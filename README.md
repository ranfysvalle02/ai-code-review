# Building an AI "Mediator" for GitHub PRs using GPT-4o and Structured Outputs

We’ve all seen it happen. A Pull Request opens. A reviewer leaves a comment. The author replies. Suddenly, a 30-message thread explodes debating the merits of a specific design pattern or variable name.

The PR stalls. Context switching becomes painful as you jump between the code diff and a wall of text.

Wouldn't it be amazing if an unbiased third party could step in, read the entire history of the argument, look at the code in question, and say: **"Here is the core disagreement, and here is the exact code change to fix it"**?

In this post, we’re going to build exactly that: an AI-powered PR mediator using Python, GitHub’s GraphQL API, and OpenAI’s GPT-4o with Structured Outputs.

-----

## The Goal

We want a script that targets a specific Pull Request branch and does the following:

1.  **Identifies all comment threads**, separating resolved from unresolved ones.
2.  **Extracts context**: The conversation history *and* the relevant code snippet (the diff hunk) where the discussion is happening.
3.  **Feeds this context to GPT-4o** with a specific persona: an expert Senior Software Engineer acting as a mediator.
4.  **Returns structured data**: A summary, severity level, reasoning, and, crucially, the *exact proposed code fix*.

## The Challenge: Getting the Right Context

The first hurdle is getting the data efficiently. GitHub's standard REST API is great, but getting comments, replies, thread status, *and* the associated diff hunk for every single thread requires way too many API calls.

This is where GraphQL shines. We can fetch everything we need in a single, precise query.

We need to query repository `pullRequests`, dive into `reviewThreads`, and grab the `diffHunk` (the actual code context) along with the comment history.

```graphql
query($owner: String!, $repo: String!, $branch: String!) {
  repository(owner: $owner, name: $repo) {
    pullRequests(headRefName: $branch, first: 1, states: OPEN) {
      nodes {
        # ... other PR details
        reviewThreads(first: 50) {
          nodes {
            isResolved
            path
            comments(first: 50) { 
              nodes {
                author { login }
                body
                diffHunk  # <--- The magic sauce: the code context
              }
            }
          }
        }
      }
    }
  }
}
```

## The "Secret Sauce": GPT-4o Structured Outputs

Asking an LLM to "fix some code" usually results in a chatty response that requires complex regex to parse. We need reliable, machine-readable JSON.

We use OpenAI's `response_format` feature, combined with **Pydantic**, to define exactly what the AI must return. If the LLM can't fit its answer into this schema, it tries again until it can.

This guarantees that our script never crashes because the AI decided to add an introductory paragraph instead of just raw JSON.

Here is the Pydantic schema we'll use:

```python
from pydantic import BaseModel, Field
from typing import Optional

class CodeSuggestion(BaseModel):
    summary: str = Field(..., description="A 1-sentence summary of the disagreement or discussion.")
    severity: str = Field(..., description="How critical is this? (Low, Medium, High)")
    # This is the magic field. It's optional. If the thread is resolved, it's null.
    # If unresolved, it contains the exact python code fix.
    proposed_fix: Optional[str] = Field(None, description="The specific Python code to resolve the issue. Null if no code change is needed.")
    reasoning: str = Field(..., description="Why this fix is the correct solution based on the conversation.")
```

## The Full Implementation

We will combine these concepts into a single Python script. It uses `requests` for the synchronous GraphQL fetch and `asyncio` with the `openai` library to process multiple threads in parallel, speeding up analysis.

Here is the complete `main.py`:

```python
import os
import json
import asyncio
import requests
from typing import Optional, List, Dict, Any
# Remember to pip install python-dotenv
from dotenv import load_dotenv

# OpenAI & Pydantic
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# --- LOAD SECRETS ---
# Expects a .env file in the same directory
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Target Repository Config
REPO_OWNER = os.getenv("GITHUB_REPO_OWNER")
REPO_NAME = os.getenv("GITHUB_REPO_NAME")
BRANCH_NAME = os.getenv("GITHUB_BRANCH_NAME")

if not all([GITHUB_TOKEN, OPENAI_API_KEY, REPO_OWNER, REPO_NAME, BRANCH_NAME]):
    raise ValueError("❌ Missing environment variables. Please check your .env file.")

# --- AI MODELS (Pydantic Schema) ---
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
        
        # The GraphQL query to get threads, comments, and diffHunks
        query = """
        query($owner: String!, $repo: String!, $branch: String!) {
          repository(owner: $owner, name: $repo) {
            pullRequests(headRefName: $branch, first: 1, states: OPEN) {
              nodes {
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
        """Cleans raw GraphQL data into a list of thread contexts."""
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

            # Find the code snippet (diffHunk) associated with this thread
            for c in thread["comments"]["nodes"]:
                if c.get("diffHunk"):
                    thread_data["code_snippet"] = c["diffHunk"]
                    break

            # Format the conversation history
            for c in thread["comments"]["nodes"]:
                msg = {
                    "author": c["author"]["login"] if c["author"] else "Unknown",
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
        """Sends a single thread context to GPT-4o for structured analysis."""
        system_prompt = (
            "You are an expert Senior Software Engineer acting as a mediator. "
            "Review the following GitHub Pull Request comment thread. "
            "Your goal is to summarize the conversation and, if unresolved, propose a code fix."
        )

        # We feed the AI the status, file path, the code diff, and the conversation history.
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
            # The magic happens here: response_format=CodeSuggestion
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
        # Process all threads simultaneously
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
                # This output is pure code, ready to copy-paste
                print(analysis.proposed_fix)
                print(f"   {'-'*30}")
        
        print("-" * 60)

# --- MAIN EXECUTION ---
async def main():
    # 1. Fetch Context from GitHub
    fetcher = GitHubFetcher(GITHUB_TOKEN, REPO_OWNER, REPO_NAME, BRANCH_NAME)
    context_data = fetcher.get_pr_context()

    if not context_data:
        print("⚠️ No context found or error occurred.")
        return

    print(f"✅ Successfully gathered {len(context_data)} threads.")

    # 2. Analyze with AI in parallel
    reviewer = AIReviewer(OPENAI_API_KEY)
    results = await reviewer.analyze_batch(context_data)

    # 3. Print Report
    print_report(results)

if __name__ == "__main__":
    asyncio.run(main())
```

## Running the Mediator

To run this, you just need a `.env` file with your credentials and target repository info:

```ini
GITHUB_TOKEN=your_github_pat
OPENAI_API_KEY=sk-your_openai_key
GITHUB_REPO_OWNER=my-org
GITHUB_REPO_NAME=my-cool-app
GITHUB_BRANCH_NAME=feature/login-fix
```

Run the script, and watch it work.

### The Output

Here is an example of what the script outputs when run against a PR with one resolved thread and one active argument about an asynchronous function:

```text
--- 🔍 Fetching Context for Branch: feature/login-fix ---
✅ Successfully gathered 2 threads.
🚀 Starting parallel AI analysis of 2 threads...

============================================================
🤖 AI AUTO-REVIEW REPORT
============================================================

✅ [RESOLVED] src/utils/logger.py
   📝 Summary: The reviewer suggested changing the log level from INFO to DEBUG, which the author accepted.
------------------------------------------------------------

🔴 [UNRESOLVED] src/api/users.py
   🔥 Severity: High
   💡 Reasoning: The current implementation uses a synchronous database call inside an async route handler, which will block the event loop and degrade performance under load. It needs to be awaited.

   🛠️ PROPOSED FIX:
   ------------------------------
   @router.get("/{user_id}")
   async def get_user(user_id: int, db: Session = Depends(get_db)):
       # Changed db.query(...) to await db.execute(...) for async compatibility
       result = await db.execute(select(User).filter(User.id == user_id))
       user = result.scalars().first()
       if user is None:
           raise HTTPException(status_code=404, detail="User not found")
       return user
   ------------------------------
------------------------------------------------------------
```

## Conclusion and Next Steps

By combining GraphQL's ability to fetch deeply nested context with GPT-4o's structured output capabilities, we've created a powerful tool that cuts through the noise of PR comment sections.

Instead of re-reading 20 comments to remember what the argument was about, you get a 1-sentence summary and the exact code needed to resolve it.

**Where could we take this next?**

1.  **GitHub Action:** Turn this script into an Action that runs automatically whenever a PR comment thread reaches a certain length.
2.  **Auto-Suggest:** Instead of just printing to the console, use the GitHub API to post the `proposed_fix` directly back to the PR as a suggested change comment.
3.  **Context Expansion:** Feed the AI related files (e.g., the unit test file corresponding to the code being modified) for even better suggestions.
