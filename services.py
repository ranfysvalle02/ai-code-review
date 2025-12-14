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
import shutil
import tempfile
import subprocess
import voyageai
import numpy as np
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, List, Dict, Any, Callable
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

@asynccontextmanager
async def git_sandbox(owner: str, repo: str, token: str, pr_number: int = None, branch: str = None, on_progress: Callable[[str], None] = None):
    """
    Clones a repo into a temporary directory and checks out the specific branch/PR.
    """
    # 1. Create a temporary directory
    sandbox_dir = tempfile.mkdtemp(prefix=f"sandbox_{owner}_{repo}_")
    
    try:
        # 2. Construct Authenticated URL (Careful with logging this!)
        clone_url = f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"
        
        if on_progress: on_progress(f"📦 Cloning repository {owner}/{repo}...")
        print(f"📦 Cloning into {sandbox_dir}...")
        
        # 3. Clone the repository (using --depth 1 for speed)
        # We use asyncio.to_thread to avoid blocking the event loop with synchronous subprocess calls
        # or we could use asyncio.create_subprocess_exec, but to_thread with subprocess.run is often simpler for sequence
        
        def run_git(args, cwd=None):
            subprocess.run(args, cwd=cwd, check=True, capture_output=True)

        await asyncio.to_thread(run_git, ["git", "clone", "--depth", "1", clone_url, sandbox_dir])
        
        if pr_number:
            if on_progress: on_progress(f"checking out PR #{pr_number}...")
            await asyncio.to_thread(run_git, ["git", "fetch", "origin", f"pull/{pr_number}/head:pr-{pr_number}"], sandbox_dir)
            await asyncio.to_thread(run_git, ["git", "checkout", f"pr-{pr_number}"], sandbox_dir)
            
        elif branch:
            if on_progress: on_progress(f"checking out branch {branch}...")
            await asyncio.to_thread(run_git, ["git", "fetch", "origin", f"{branch}:{branch}"], sandbox_dir)
            await asyncio.to_thread(run_git, ["git", "checkout", branch], sandbox_dir)

        yield sandbox_dir

    except subprocess.CalledProcessError as e:
        print(f"❌ Git Operation Failed: {e.stderr.decode() if e.stderr else 'Unknown Error'}")
        if on_progress: on_progress(f"❌ Git failed: {e.stderr.decode() if e.stderr else 'Unknown Error'}")
        raise e
    except Exception as e:
         if on_progress: on_progress(f"❌ Error: {str(e)}")
         raise e
    finally:
        # 4. Cleanup
        if os.path.exists(sandbox_dir):
            await asyncio.to_thread(shutil.rmtree, sandbox_dir)
            print(f"🧹 Cleaned up {sandbox_dir}")

# --- AI MODELS ---

class VectorStore:
    def __init__(self, api_key: str):
        self.client = voyageai.Client(api_key=api_key)
        self.documents = [] 
        self.embeddings = [] 
        self.metadata = [] 

    async def index_repo(self, root_dir: str, on_progress: Callable[[str], None] = None, include_paths: List[str] = None):
        # Run the heavy indexing logic in a separate thread to keep the event loop responsive
        return await asyncio.to_thread(self._index_repo_sync, root_dir, on_progress, include_paths)

    def _index_repo_sync(self, root_dir: str, on_progress: Callable[[str], None] = None, include_paths: List[str] = None):
        if on_progress: on_progress("🔍 Scanning files for indexing...")
        print(f"🔍 Indexing repo at {root_dir}...")
        if include_paths:
            print(f"🎯 Scoping index to: {include_paths}")
            
        chunks = []
        meta = []
        
        total_files = 0
        
        # Walk through files
        for root, _, files in os.walk(root_dir):
            if ".git" in root: continue
            
            # Comprehensive list of code and config extensions
            code_extensions = (
                # Python & Data Science
                ".py", ".pyw", ".pyi", ".ipynb", ".r", ".R", ".jl",
                # Web (JS/TS/HTML/CSS/Frameworks)
                ".js", ".mjs", ".cjs", ".ts", ".mts", ".cts", ".tsx", ".jsx", 
                ".html", ".htm", ".css", ".scss", ".sass", ".less", 
                ".vue", ".svelte", ".astro", ".php", ".php4", ".php5", ".phtml",
                # System & High Performance
                ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".c++", ".hh", 
                ".rs", ".rlib", ".go", ".zig", ".s", ".asm", ".wat", ".wasm",
                # JVM
                ".java", ".kt", ".kts", ".scala", ".sc", ".groovy", ".clj", ".cljs", ".cljc",
                # Apple
                ".swift", ".m", ".mm",
                # Ruby & Crystal
                ".rb", ".erb", ".rake", ".cr",
                # Erlang/Elixir
                ".ex", ".exs", ".erl", ".hrl",
                # .NET
                ".cs", ".fs", ".fsx", ".fsi", ".vb",
                # Functional (Haskell, OCaml, etc)
                ".hs", ".lhs", ".ml", ".mli", ".elm", ".purs",
                # Shell & Scripting
                ".sh", ".bash", ".zsh", ".fish", ".ps1", ".psm1", ".bat", ".cmd", ".lua", ".pl", ".pm", ".t",
                # Mobile / Other
                ".dart", ".sol",
                # Config / Infra / IDL
                ".sql", ".graphql", ".gql", ".proto", ".tf", ".tfvars", 
                ".yaml", ".yml", ".json", ".toml", ".xml", ".ini", ".conf",
                ".dockerfile", "Dockerfile", "Makefile", "Justfile", "Rakefile", "Gemfile", "Cargo.toml", "go.mod", "package.json"
            )

            for file in files:
                # Case-insensitive check for extensions and exact filenames
                if not file.lower().endswith(tuple(ext.lower() for ext in code_extensions)):
                    continue
                
                # Check path filtering
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_dir)
                
                if include_paths:
                    # Check if this file starts with any of the include_paths
                    # (include_paths can be "src/api" matching "src/api/routes.py" or "src/config.py" exact match)
                    if not any(rel_path.startswith(p) for p in include_paths):
                        continue

                total_files += 1
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    lines = content.split("\n")
                    chunk_size = 50
                    for i in range(0, len(lines), chunk_size):
                        chunk_text = "\n".join(lines[i:i+chunk_size])
                        if not chunk_text.strip(): continue
                        
                        chunks.append(chunk_text)
                        meta.append({
                            "file_path": os.path.relpath(file_path, root_dir),
                            "line_start": i + 1,
                            "line_end": min(i + chunk_size, len(lines))
                        })
                except Exception as e:
                    pass

        if not chunks:
            if on_progress: on_progress("⚠️ No code found to index.")
            return {"total_files": total_files, "total_chunks": 0}

        if on_progress: on_progress(f"📊 Embedding {len(chunks)} code chunks from {total_files} files with Voyage AI...")
        print(f"📊 Embedding {len(chunks)} chunks from {total_files} files with Voyage AI...")
        
        # Batch embedding
        batch_size = 128
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            if on_progress and i > 0: on_progress(f"📊 Processed {i}/{len(chunks)} chunks...")
            batch = chunks[i:i+batch_size]
            result = self.client.embed(batch, model="voyage-code-2")
            all_embeddings.extend(result.embeddings)
            
        self.documents = chunks
        self.embeddings = np.array(all_embeddings)
        self.metadata = meta
        
        stats = {"total_files": total_files, "total_chunks": len(chunks)}
        if on_progress: on_progress(f"✅ Indexing complete. ({total_files} files, {len(chunks)} chunks)")
        print("✅ Indexing complete.")
        return stats

    def search(self, query: str, k: int = 3):
        if len(self.embeddings) == 0:
            return []
            
        query_emb = self.client.embed([query], model="voyage-code-2").embeddings[0]
        query_vec = np.array(query_emb)
        
        # Cosine similarity
        # (Assuming embeddings are normalized, which Voyage usually does, but let's be safe)
        norm_query = np.linalg.norm(query_vec)
        norm_docs = np.linalg.norm(self.embeddings, axis=1)
        
        similarities = np.dot(self.embeddings, query_vec) / (norm_docs * norm_query)
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "text": self.documents[idx],
                "metadata": self.metadata[idx],
                "score": float(similarities[idx])
            })
            
        return results

class CodeSuggestion(BaseModel):
    summary: str = Field(..., description="A 1-sentence summary of the disagreement or discussion.")
    severity: str = Field(..., description="How critical is this? (Low, Medium, High)")
    thinking_process: List[str] = Field(..., description="A step-by-step list of your internal reasoning process. Explain how you analyzed the code, the context, and the arguments to reach your conclusion.")
    code_analysis: str = Field(..., description="Detailed technical analysis of the code snippets, identifying patterns, bugs, or performance issues.")
    debate_analysis: str = Field(..., description="Analysis of the arguments presented by the reviewers and the author.")
    resolution_critique: str = Field(..., description="Critique of how the issue was resolved. Was it a proper fix or a workaround?")
    alternative_approaches: List[str] = Field(..., description="List of alternative technical approaches that could have been taken.")
    proposed_fix: Optional[str] = Field(None, description="The specific Python code to resolve the issue. Null if no code change is needed.")
    fix_relevance: Optional[str] = Field(None, description="A short snippet or quote from the conversation that justifies this specific fix (e.g. 'Addressing @user's concern about race conditions').")
    reasoning: str = Field(..., description="Why this fix is the correct solution based on the conversation.")

class GitHubFetcher:
    def __init__(self, token):
        self.headers = {"Authorization": f"Bearer {token}"}
        self.url = "https://api.github.com/graphql"

    def list_prs(self, owner: str, repo: str, states: List[str] = ["OPEN", "CLOSED", "MERGED"], cursor: str = None) -> Optional[Dict[str, Any]]:
        print(f"--- 🔍 Fetching PRs for {owner}/{repo} {states} (cursor={cursor}) ---")
        query = """
        query($owner: String!, $repo: String!, $states: [PullRequestState!], $cursor: String) {
          repository(owner: $owner, name: $repo) {
            pullRequests(first: 100, states: $states, after: $cursor, orderBy: {field: CREATED_AT, direction: DESC}) {
              pageInfo {
                endCursor
                hasNextPage
              }
              nodes {
                number
                title
                state
                headRefName
                createdAt
                author {
                  login
                }
              }
            }
          }
        }
        """
        variables = {"owner": owner, "repo": repo, "states": states, "cursor": cursor}
        response = requests.post(self.url, json={"query": query, "variables": variables}, headers=self.headers)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}\n{response.text}")
            return None

        data = response.json()
        if "errors" in data:
            print("❌ GraphQL Error:", data["errors"])
            return None

        try:
            return data["data"]["repository"]["pullRequests"]
        except KeyError as e:
            print(f"❌ Error parsing response: {e}")
            return None

    def search_repositories(self, query: str, first: int = 10) -> Optional[List[Dict[str, Any]]]:
        print(f"--- 🔍 Searching Repositories for '{query}' ---")
        gql_query = """
        query($query: String!, $first: Int!) {
          search(query: $query, type: REPOSITORY, first: $first) {
            nodes {
              ... on Repository {
                name
                owner { login }
                description
                stargazerCount
                forkCount
                primaryLanguage { name }
                updatedAt
                url
              }
            }
          }
        }
        """
        variables = {"query": query, "first": first}
        response = requests.post(self.url, json={"query": gql_query, "variables": variables}, headers=self.headers)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}\n{response.text}")
            return None

        data = response.json()
        if "errors" in data:
            print("❌ GraphQL Error:", data["errors"])
            return None

        try:
            return data["data"]["search"]["nodes"]
        except KeyError as e:
            print(f"❌ Error parsing response: {e}")
            return None

    def search_prs(self, owner: str, repo: str, search_term: str, cursor: str = None) -> Optional[Dict[str, Any]]:
        print(f"--- 🔍 Searching PRs for {owner}/{repo} term='{search_term}' (cursor={cursor}) ---")
        # Construct search query: repo:owner/repo is:pr is:open term
        search_query = f"repo:{owner}/{repo} is:pr is:open {search_term}"
        
        query = """
        query($search_query: String!, $cursor: String) {
          search(query: $search_query, type: ISSUE, first: 100, after: $cursor) {
            pageInfo {
              endCursor
              hasNextPage
            }
            nodes {
              ... on PullRequest {
                number
                title
                state
                headRefName
                createdAt
                author {
                  login
                }
              }
            }
          }
        }
        """
        variables = {"search_query": search_query, "cursor": cursor}
        response = requests.post(self.url, json={"query": query, "variables": variables}, headers=self.headers)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}\n{response.text}")
            return None

        data = response.json()
        if "errors" in data:
            print("❌ GraphQL Error:", data["errors"])
            return None

        try:
            return data["data"]["search"]
        except KeyError as e:
            print(f"❌ Error parsing response: {e}")
            return None

    def get_pr_threads(self, owner: str, repo: str, pr_number: int) -> Optional[List[Dict[str, Any]]]:
        print(f"--- 🔍 Fetching Threads for PR #{pr_number} in {owner}/{repo} ---")
        query = """
        query($owner: String!, $repo: String!, $pr_number: Int!) {
          repository(owner: $owner, name: $repo) {
            pullRequest(number: $pr_number) {
              title
              reviewThreads(first: 50) {
                nodes {
                  id
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
        """
        variables = {"owner": owner, "repo": repo, "pr_number": pr_number}
        response = requests.post(self.url, json={"query": query, "variables": variables}, headers=self.headers)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}\n{response.text}")
            return None

        data = response.json()
        if "errors" in data:
            print("❌ GraphQL Error:", data["errors"])
            return None

        try:
            pr_data = data["data"]["repository"]["pullRequest"]
            if not pr_data:
                 print(f"❌ PR #{pr_number} not found.")
                 return None
            
            threads = pr_data["reviewThreads"]["nodes"]
            return self._parse_threads(threads)
        except KeyError as e:
            print(f"❌ Error parsing response: {e}")
            return None

    def get_file_tree(self, owner: str, repo: str, branch: str = "HEAD") -> List[str]:
        print(f"--- 🔍 Fetching File Tree for {owner}/{repo} ---")
        # Use REST API for recursive tree
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}\n{response.text}")
            return []

        data = response.json()
        try:
            # Filter for blobs (files) only, or include trees if needed. 
            # Let's return all paths so AI can select dirs or files.
            return [item["path"] for item in data.get("tree", [])]
        except (KeyError, TypeError) as e:
            print(f"❌ Error parsing tree: {e}")
            return []

    def get_repo_structure(self, owner: str, repo: str) -> Optional[List[str]]:
        print(f"--- 🔍 Fetching Repo Structure for {owner}/{repo} ---")
        query = """
        query($owner: String!, $repo: String!) {
          repository(owner: $owner, name: $repo) {
            object(expression: "HEAD:") {
              ... on Tree {
                entries {
                  name
                  type
                }
              }
            }
          }
        }
        """
        variables = {"owner": owner, "repo": repo}
        response = requests.post(self.url, json={"query": query, "variables": variables}, headers=self.headers)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}\n{response.text}")
            return None

        data = response.json()
        try:
            entries = data["data"]["repository"]["object"]["entries"]
            # Return only directories
            return [e["name"] for e in entries if e["type"] == "tree"]
        except (KeyError, TypeError) as e:
            print(f"❌ Error parsing structure: {e}")
            return []

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> List[str]:
        print(f"--- 🔍 Fetching Files for PR #{pr_number} ---")
        query = """
        query($owner: String!, $repo: String!, $pr_number: Int!) {
          repository(owner: $owner, name: $repo) {
            pullRequest(number: $pr_number) {
              files(first: 100) {
                nodes {
                  path
                }
              }
            }
          }
        }
        """
        variables = {"owner": owner, "repo": repo, "pr_number": pr_number}
        response = requests.post(self.url, json={"query": query, "variables": variables}, headers=self.headers)
        
        if response.status_code != 200:
            return []

        data = response.json()
        try:
            nodes = data["data"]["repository"]["pullRequest"]["files"]["nodes"]
            return [n["path"] for n in nodes]
        except (KeyError, TypeError):
            return []

    def _parse_threads(self, threads):
        llm_context = []
        for thread in threads:
            if not thread["comments"]["nodes"]:
                continue

            thread_data = {
                "id": thread["id"],
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

class SuggestedContext(BaseModel):
    reasoning: str = Field(..., description="Why these paths were selected based on the PR changes.")
    selected_paths: List[str] = Field(..., description="List of specific file paths or directory paths to index.")

class AIReviewer:
    def __init__(self, api_key):
        self.client = AsyncOpenAI(api_key=api_key)

    async def suggest_indexing_paths(self, file_tree: List[str], pr_files: List[str]) -> SuggestedContext:
        # Smart Context Strategy:
        # 1. Markdown Files: High value for understanding architecture/intent.
        # 2. Root Files: Config/Project definitions.
        # 3. Directories: Structural context without overwhelming token limits.
        
        # Filter markdown and text documentation
        md_files = [p for p in file_tree if p.lower().endswith(('.md', '.mdx', '.txt', '.rst'))]
        
        # Filter root files (no directory separators)
        root_files = [p for p in file_tree if '/' not in p]
        
        # Extract unique directories from the rest
        directories = set()
        for path in file_tree:
            dirname = os.path.dirname(path)
            if dirname and dirname != ".":
                # Add all parent directories
                parts = dirname.split('/')
                for i in range(len(parts)):
                    directories.add('/'.join(parts[:i+1]))
        
        sorted_dirs = sorted(list(directories))
        
        # Construct the context list
        # We prioritize MD files and Root files, then fill remaining space with directories
        final_list = []
        
        # Add Root files (Critical config)
        final_list.extend(sorted(root_files))
        
        # Add Markdown files (Docs) - Cap at 500 to ensure good coverage
        final_list.extend(sorted(md_files)[:500]) 
        
        # Add Directories (Structure)
        final_list.extend(sorted_dirs)
        
        # Deduplicate
        final_list = sorted(list(set(final_list)))
        
        # Hard Cap to prevent 429 errors (Token Limit ~30k)
        # 2000 paths * ~10 tokens/path = ~20k tokens. Safe zone.
        MAX_ITEMS = 2000
        if len(final_list) > MAX_ITEMS:
            print(f"⚠️ Truncating context structure ({len(final_list)} -> {MAX_ITEMS})")
            final_list = final_list[:MAX_ITEMS]

        prompt = (
            "You are an expert software architect. A user is reviewing a Pull Request and needs to decide which specific files or folders to index for RAG context.\n"
            "The goal is to select ONLY the code that is relevant to the changes to keep the context focused and fast.\n\n"
            f"PR MODIFIED FILES: {json.dumps(pr_files)}\n\n"
            f"REPO STRUCTURE (Smart Digest): {json.dumps(final_list)}\n\n"
            "INSTRUCTIONS:\n"
            "1. Analyze the PR modified files. Identify the modules they belong to.\n"
            "2. Review the REPO STRUCTURE. It contains specific documentation files (md/txt), root config files, and directory paths.\n"
            "3. Select specific directories (e.g. `src/api`) that likely contain relevant dependencies or logic.\n"
            "4. **CRITICAL:** Select relevant documentation files (e.g. `docs/ARCHITECTURE.md`, `README.md`) that might explain the modified components.\n"
            "5. Return the list of selected paths."
        )
        
        try:
            completion = await self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": prompt}],
                response_format=SuggestedContext,
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"❌ Error getting context suggestions: {e}")
            return SuggestedContext(reasoning="Error or rate limit. Defaulting to empty context.", selected_paths=[])

    async def analyze_thread(self, thread_data, sandbox_dir: str, custom_prompt: str = None, retriever: VectorStore = None):
        default_system_prompt = (
            "You are an expert Senior Software Engineer acting as a technical mediator and auditor. "
            "Your goal is to provide a deep, 'transparent box' analysis of the following GitHub Pull Request thread.\n\n"
            "DO NOT just summarize what people said. You must:\n"
            "1. Analyze the Code: Look for bugs, race conditions, bad patterns, or performance bottlenecks in the provided snippet and RAG context.\n"
            "2. Evaluate the Debate: Assess the technical merit of each argument using first principles. Who is right, and why?\n"
            "3. Critique the Resolution: Even if the thread is 'RESOLVED', did they actually fix it correctly? Or did they just merge a band-aid?\n"
            "4. Propose Alternatives: What other ways could this have been solved? Are there better patterns?\n"
            "5. Show Your Work: Fill the 'thinking_process' field with your step-by-step investigation.\n"
            "   - 'code_analysis': specific technical deep dive.\n"
            "   - 'debate_analysis': weigh the arguments.\n"
            "   - 'resolution_critique': honestly evaluate the outcome.\n"
            "   - 'alternative_approaches': list 2-3 viable alternatives.\n"
            "   - 'fix_relevance': quote the specific user concern this fix addresses."
        )
        
        system_prompt = default_system_prompt
        
        # --- 1. Get Full Content of Target File ---
        target_file_content = ""
        full_file_path = os.path.join(sandbox_dir, thread_data['file_path'])
        try:
            if os.path.exists(full_file_path):
                with open(full_file_path, "r", encoding="utf-8") as f:
                    target_file_content = f.read()
            else:
                target_file_content = "(File not found in sandbox - possibly deleted or renamed in this PR)"
        except Exception as e:
            target_file_content = f"(Error reading file: {e})"

        # --- 1.5 Get Project Guidelines (Contributing, etc.) ---
        guidelines_content = ""
        guidelines_files = ["CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "STYLE.md", "DEVELOPMENT.md"]
        found_guidelines = []
        
        for g_file in guidelines_files:
            # Check root
            g_path = os.path.join(sandbox_dir, g_file)
            if not os.path.exists(g_path):
                # Check .github folder
                g_path = os.path.join(sandbox_dir, ".github", g_file)
            
            if os.path.exists(g_path):
                try:
                    with open(g_path, "r", encoding="utf-8") as f:
                        # Truncate to avoid blowing context if huge
                        content = f.read(5000) 
                        found_guidelines.append(f"--- {g_file} ---\n{content}\n")
                except:
                    pass
        
        if found_guidelines:
            guidelines_content = "\n--- 📜 PROJECT GUIDELINES & STANDARDS ---\n" + "\n".join(found_guidelines)

        # --- 2. RAG: Retrieve Related Context ---
        rag_context = ""
        rag_sources = [] 
        if retriever:
            # Smart Query: Combine file path, diff, and the last comment (usually the most relevant context)
            last_comment = thread_data['conversation'][-1]['text'] if thread_data['conversation'] else ""
            query_text = f"File: {thread_data['file_path']}\nDiff: {thread_data['code_snippet']}\nContext: {last_comment}"
            
            # Search
            relevant_chunks = retriever.search(query_text, k=5) # Increased k for better coverage
            if relevant_chunks:
                rag_context = "\n--- 📚 RELATED CODE (Dependencies/Usage) ---\n"
                for i, chunk in enumerate(relevant_chunks):
                    meta = chunk['metadata']
                    
                    # Optimization: Skip chunks that are just parts of the target file we already read fully
                    if meta['file_path'] == thread_data['file_path']:
                        continue

                    # Structure the source for UI display
                    rag_sources.append({
                        "file_path": meta['file_path'],
                        "line_start": meta['line_start'],
                        "line_end": meta['line_end'],
                        "content": chunk['text']
                    })
                    rag_context += f"\n[Context {i+1}] {meta['file_path']} (Lines {meta['line_start']}-{meta['line_end']}):\n"
                    rag_context += chunk['text'] + "\n"
        
        if custom_prompt:
             system_prompt += f"\n\nADDITIONAL INSTRUCTIONS/QUESTIONS:\n{custom_prompt}"

        user_content = f"""
        STATUS: {thread_data['status']}
        FILE: {thread_data['file_path']}
        
        {guidelines_content}
        
        --- 📄 TARGET FILE CONTENT (Full) ---
        {target_file_content}
        
        {rag_context}
        
        --- 📝 CODE SNIPPET (The Diff Under Discussion) ---
        {thread_data.get('code_snippet', '(No code snippet provided)')}
        
        --- 💬 CONVERSATION HISTORY ---
        {json.dumps(thread_data['conversation'], indent=2)}
        
        INSTRUCTIONS:
        1. If STATUS is 'RESOLVED', verify the fix is robust. If it is, set 'proposed_fix' to null. If it's weak, propose a better one.
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
                "id": thread_data.get("id"),
                "file": thread_data["file_path"],
                "status": thread_data["status"],
                "code_snippet": thread_data.get("code_snippet"),
                "rag_sources": rag_sources,
                "ai_analysis": completion.choices[0].message.parsed
            }
        except Exception as e:
            print(f"❌ Error analyzing {thread_data['file_path']}: {e}")
            return None

    async def analyze_batch(self, threads, sandbox_dir: str, custom_prompt: str = None, retriever: VectorStore = None, on_progress: Callable[[str], None] = None):
        print(f"🚀 Starting parallel AI analysis of {len(threads)} threads...")
        if on_progress: on_progress(f"🚀 Launching {len(threads)} parallel AI agents...")
        
        async def analyze_wrapper(t):
            res = await self.analyze_thread(t, sandbox_dir, custom_prompt, retriever)
            if on_progress: on_progress(f"✅ Analyzed thread: {t['file_path']}")
            return res

        tasks = [analyze_wrapper(t) for t in threads]
        return await asyncio.gather(*tasks)
