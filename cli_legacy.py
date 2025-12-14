# Copyright (C) 2025 Fabian Valle-simmons
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import json
import asyncio
import requests
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# OpenAI & Pydantic
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# --- LOAD SECRETS ---
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

REPO_OWNER = os.getenv("GITHUB_REPO_OWNER")
REPO_NAME = os.getenv("GITHUB_REPO_NAME")
BRANCH_NAME = os.getenv("GITHUB_BRANCH_NAME")

if not all([GITHUB_TOKEN, OPENAI_API_KEY, REPO_OWNER, REPO_NAME, BRANCH_NAME]):
    raise ValueError("❌ Missing environment variables. Please check your .env file.")

# --- AI MODELS ---
class CodeSuggestion(BaseModel):
    summary: str = Field(..., description="A 1-sentence summary of the disagreement or discussion.")
    severity: str = Field(..., description="How critical is this? (Low, Medium, High)")
    proposed_fix: Optional[str] = Field(None, description="The specific Python code to resolve the issue. Null if no code change is needed.")
    reasoning: str = Field(..., description="Why this fix is the correct solution based on the conversation.")

# --- CLASS: GITHUB FETCHER ---
class GitHubFetcher:
    def __init__(self, token, owner, repo, branch):
        self.headers = {"Authorization": f"Bearer {token}"}
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.url = "https://api.github.com/graphql"

    def get_pr_context(self) -> Optional[List[Dict[str, Any]]]:
        print(f"--- 🔍 Fetching Context for Branch: {self.branch} ---")
        
        query = """
        query($owner: String!, $repo: String!, $branch: String!) {
          repository(owner: $owner, name: $repo) {
            pullRequests(headRefName: $branch, first: 1, states: OPEN) {
              nodes {
                number
                title
                reviewThreads(first: 50) {
                  nodes {
                    isResolved
                    path
                    comments(first: 50) { 
                      nodes {
                        author { login }
                        body
                        createdAt
                        diffHunk 
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        variables = {"owner": self.owner, "repo": self.repo, "branch": self.branch}
        response = requests.post(self.url, json={"query": query, "variables": variables}, headers=self.headers)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}\n{response.text}")
            return None

        data = response.json()
        if "errors" in data:
            print("❌ GraphQL Error:", data["errors"])
            return None

        try:
            pr_list = data["data"]["repository"]["pullRequests"]["nodes"]
            if not pr_list:
                print(f"❌ No open PR found for branch: {self.branch}")
                return None

            threads = pr_list[0]["reviewThreads"]["nodes"]
            return self._parse_threads(threads)

        except KeyError as e:
            print(f"❌ Error parsing response: {e}")
            return None

    def _parse_threads(self, threads):
        llm_context = []
        for thread in threads:
            if not thread["comments"]["nodes"]:
                continue

            thread_data = {
                "file_path": thread["path"],
                "status": "RESOLVED" if thread["isResolved"] else "UNRESOLVED",
                "code_snippet": "", 
                "conversation": []
            }

            # Extract diff hunk
            for c in thread["comments"]["nodes"]:
                if c.get("diffHunk"):
                    thread_data["code_snippet"] = c["diffHunk"]
                    break

            # Extract conversation
            for c in thread["comments"]["nodes"]:
                msg = {
                    "author": c["author"]["login"] if c["author"] else "Unknown",
                    "timestamp": c["createdAt"],
                    "text": c["body"]
                }
                thread_data["conversation"].append(msg)

            llm_context.append(thread_data)
        
        return llm_context

# --- CLASS: AI REVIEWER ---
class AIReviewer:
    def __init__(self, api_key):
        self.client = AsyncOpenAI(api_key=api_key)

    async def analyze_thread(self, thread_data):
        system_prompt = (
            "You are an expert Senior Software Engineer acting as a mediator. "
            "Review the following GitHub Pull Request comment thread. "
            "Your goal is to summarize the conversation and, if unresolved, propose a code fix."
        )

        user_content = f"""
        STATUS: {thread_data['status']}
        FILE: {thread_data['file_path']}
        
        --- CODE SNIPPET (DIFF) ---
        {thread_data.get('code_snippet', '(No code snippet provided)')}
        
        --- CONVERSATION ---
        {json.dumps(thread_data['conversation'], indent=2)}
        
        INSTRUCTIONS:
        1. If STATUS is 'RESOLVED', just summarize the discussion. Set 'proposed_fix' to null.
        2. If STATUS is 'UNRESOLVED', summarize the blocking issue and write the EXACT code change needed to fix it in 'proposed_fix'.
        """

        try:
            completion = await self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format=CodeSuggestion,
            )
            return {
                "file": thread_data["file_path"],
                "status": thread_data["status"],
                "ai_analysis": completion.choices[0].message.parsed
            }
        except Exception as e:
            print(f"❌ Error analyzing {thread_data['file_path']}: {e}")
            return None

    async def analyze_batch(self, threads):
        print(f"🚀 Starting parallel AI analysis of {len(threads)} threads...")
        tasks = [self.analyze_thread(t) for t in threads]
        return await asyncio.gather(*tasks)

# --- REPORT GENERATOR ---
def print_report(results):
    print("\n" + "="*60)
    print("🤖 AI AUTO-REVIEW REPORT")
    print("="*60)

    for res in results:
        if not res: continue
        
        analysis = res['ai_analysis']
        icon = "✅" if res['status'] == "RESOLVED" else "🔴"
        
        print(f"\n{icon} [{res['status']}] {res['file']}")
        print(f"   📝 Summary: {analysis.summary}")
        
        if res['status'] == "UNRESOLVED":
            print(f"   🔥 Severity: {analysis.severity}")
            print(f"   💡 Reasoning: {analysis.reasoning}")
            
            if analysis.proposed_fix:
                print(f"\n   🛠️ PROPOSED FIX:\n   {'-'*30}")
                print(analysis.proposed_fix)
                print(f"   {'-'*30}")
        
        print("-" * 60)

# --- MAIN EXECUTION ---
async def main():
    # 1. Fetch Context
    fetcher = GitHubFetcher(GITHUB_TOKEN, REPO_OWNER, REPO_NAME, BRANCH_NAME)
    context_data = fetcher.get_pr_context()

    if not context_data:
        print("⚠️ No context found or error occurred.")
        return

    print(f"✅ Successfully gathered {len(context_data)} threads.")
    
    # Optional: Save context dump for debugging
    with open("pr_context.json", "w") as f:
        json.dump(context_data, f, indent=2)

    # 2. Analyze with AI
    reviewer = AIReviewer(OPENAI_API_KEY)
    results = await reviewer.analyze_batch(context_data)

    # 3. Print Report
    print_report(results)

if __name__ == "__main__":
    asyncio.run(main())
