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
import re
import asyncio
import time
import requests
import shutil
import tempfile
import subprocess
import voyageai
import numpy as np
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, List, Dict, Any, Callable
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, AsyncAzureOpenAI

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
        self._cache = None  # Will be set via set_cache() method
    
    def set_cache(self, cache):
        """Set the cache instance for embedding caching"""
        self._cache = cache 

    async def index_repo(self, root_dir: str, on_progress: Callable[[str], None] = None, 
                        include_paths: List[str] = None, owner: str = None, repo: str = None,
                        pr_number: int = None, commit_sha: str = None):
        """
        Index repository with optional caching.
        
        Args:
            root_dir: Root directory to index
            on_progress: Progress callback
            include_paths: Optional list of paths to include
            owner: Repository owner (for caching)
            repo: Repository name (for caching)
            pr_number: PR number (for caching)
            commit_sha: Commit SHA (for caching)
        """
        # Check cache first if we have all required parameters
        if self._cache and owner and repo and pr_number and commit_sha:
            cached = self._cache.get_embeddings(owner, repo, pr_number, commit_sha, include_paths)
            if cached:
                if on_progress: on_progress("✅ Loading embeddings from cache...")
                self.documents = cached.get("documents", [])
                self.embeddings = cached.get("embeddings", np.array([]))
                self.metadata = cached.get("metadata", [])
                
                stats = {
                    "total_files": cached.get("total_files", 0),
                    "total_chunks": len(self.documents)
                }
                if on_progress: on_progress(f"✅ Loaded {len(self.documents)} chunks from cache")
                return stats
        
        # Run the heavy indexing logic in a separate thread to keep the event loop responsive
        return await asyncio.to_thread(self._index_repo_sync, root_dir, on_progress, include_paths, 
                                       owner, repo, pr_number, commit_sha)

    def _index_repo_sync(self, root_dir: str, on_progress: Callable[[str], None] = None, 
                         include_paths: List[str] = None, owner: str = None, repo: str = None,
                         pr_number: int = None, commit_sha: str = None):
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
        
        # Store in cache if available
        if self._cache and owner and repo and pr_number and commit_sha:
            try:
                self._cache.store_embeddings(
                    owner, repo, pr_number, commit_sha,
                    documents=chunks,
                    embeddings=self.embeddings,
                    metadata=meta,
                    indexing_paths=include_paths
                )
                # Also store metadata
                self._cache.store_embedding_metadata(
                    owner, repo, pr_number, commit_sha,
                    indexing_paths=include_paths,
                    total_files=total_files,
                    total_chunks=len(chunks)
                )
            except Exception as e:
                print(f"⚠️ Warning: Failed to cache embeddings: {e}")
        
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

class FileSummary(BaseModel):
    file: str = Field(..., description="The full file path relative to the repository root.")
    description: str = Field(..., description="A concise, specific 1-2 sentence summary of WHAT changed in this file and WHY it matters. Be concrete: 'Adds MongoDB connection pooling with max 10 connections' not 'Updates database code'.")

class SpecializedAnalysis(BaseModel):
    """Results from a specialized analysis focus area"""
    focus_area: str = Field(..., description="The focus area (e.g., 'Security', 'Performance', 'Architecture')")
    findings: List[str] = Field(default=[], description="Specific findings with file paths and line numbers. Format: 'In `file.py` line X: [finding]'")
    critical_issues: List[str] = Field(default=[], description="Critical issues found in this area. Empty if none.")
    recommendations: List[str] = Field(default=[], description="Specific recommendations for this area")
    score: int = Field(..., description="Score 1-10 for this specific area")
    summary: str = Field(..., description="Brief summary of findings in this area")

class Issue(BaseModel):
    """A structured issue with description and suggested fix"""
    description: str = Field(..., description="Specific issue description with file path and line reference. Format: 'In `file.py` around line X: [specific issue]. Impact: [what happens]'. Only include genuine bugs, security flaws, or architectural problems.")
    file_path: str = Field(..., description="The file path where the issue exists (e.g., 'holly_api/api/routers/sa_sizing/apc/__init__.py')")
    line_number: Optional[int] = Field(None, description="Approximate line number where the issue occurs")
    severity: str = Field(..., description="Severity level: Critical, Important, or Medium")
    suggested_fix: str = Field(..., description="A complete, copy-pasteable code snippet showing the suggested fix. Include enough context (imports, function signature, etc.) to make it immediately usable. Provide PURE CODE ONLY - NO markdown code fences (no ```python or ```).")
    rationale: Optional[str] = Field(None, description="Brief explanation of why this fix is needed and what it improves")

class Recommendation(BaseModel):
    """A structured recommendation with code examples"""
    description: str = Field(..., description="Actionable recommendation description with file path and line reference. Format: '[Severity] In `file.py` line X: [issue description]'")
    file_path: str = Field(..., description="The file path where the recommendation applies (e.g., 'holly_api/api/models/sa_sizing.py')")
    line_number: Optional[int] = Field(None, description="Approximate line number where the change should be made")
    severity: str = Field(..., description="Severity level: Critical, Important, or Nitpick")
    current_code: Optional[str] = Field(None, description="The current code snippet that needs improvement (if applicable). Provide PURE CODE ONLY - NO markdown code fences.")
    suggested_code: str = Field(..., description="A complete, copy-pasteable code snippet showing the suggested improvement. Include enough context to make it immediately usable. Provide PURE CODE ONLY - NO markdown code fences (no ```python or ```).")
    rationale: str = Field(..., description="Explanation of why this change matters and what it improves")

class GeneralReview(BaseModel):
    summary: str = Field(..., description="A comprehensive executive summary (3-5 sentences) covering: what the PR accomplishes, overall code quality, key strengths, notable concerns, and merge readiness. Be specific and concrete.")
    overall_severity: str = Field(..., description="Overall risk level (Low, Medium, High). Low = ready to merge, Medium = needs minor fixes, High = significant issues.")
    key_issues: List[Issue] = Field(..., description="List of specific issues with file paths, line references, and suggested fixes. Each issue must include a complete, copy-pasteable code snippet showing the fix. Only include genuine bugs, security flaws, or architectural problems.")
    code_quality_score: int = Field(..., description="A score from 1-10 rating the code quality. Must be justified in detailed_analysis with specific examples of what earns points and what prevents a perfect score.")
    recommendations: List[Recommendation] = Field(..., description="Actionable recommendations with file paths, line numbers, current code (if applicable), suggested code snippets, and rationale. Each recommendation must include a complete, copy-pasteable code snippet. Include at least 2-3 Nitpick suggestions.")
    detailed_analysis: str = Field(..., description="Comprehensive technical analysis (500+ words) covering: specific code patterns observed (with file refs), comparison with existing codebase patterns, technical trade-offs, edge cases, performance implications, and explicit quality score justification explaining what earns points and what prevents a perfect score.")
    file_summaries: List[FileSummary] = Field(..., description="A concise (1-2 sentence) description for EACH modified file, focusing on WHAT changed and WHY it matters. Be specific about functionality, not generic.")
    specialized_analyses: List[SpecializedAnalysis] = Field(default=[], description="Results from parallel specialized analyses")

class ActionItem(BaseModel):
    file: str = Field(..., description="The file path associated with this action item.")
    url: Optional[str] = Field(None, description="A computed link to the file/code in the PR.")
    priority: str = Field(..., description="Priority level: High, Medium, or Low.")
    description: str = Field(..., description="A clear, concise description of the unresolved issue.")
    suggested_fix_snippet: str = Field(..., description="A short code snippet representing the suggested fix.")

class AnalysisSummary(BaseModel):
    overview: str = Field(..., description="A high-level overview of the PR's resolution status based on all analyzed threads.")
    action_items: List[ActionItem] = Field(..., description="A list of specific, unresolved issues that need attention.")
    resolution_audit: str = Field(..., description="An audit of how well the resolved threads were handled. Were fixes robust?")
    impact_tags: List[str] = Field(..., description="3-5 tags categorizing the impact (e.g., 'Performance', 'Security', 'Refactoring').")
    merge_recommendation: str = Field(..., description="Clear guidance on whether to merge, request changes, or block. (e.g., 'Ready to Merge', 'Request Changes', 'Blocked').")
    proposed_comments: List[str] = Field(..., description="Draft comments that are direct, technical, and actionable. MUST include code snippets or specific variable names. Avoid vague suggestions.")

class GitHubFetcher:
    def __init__(self, token):
        self.headers = {"Authorization": f"Bearer {token}"}
        self.url = "https://api.github.com/graphql"
        self._current_user = None
    
    def _make_request_with_retry(self, query: str, variables: Dict[str, Any] = None, 
                                 max_retries: int = 3, on_progress: Callable[[str], None] = None) -> Optional[requests.Response]:
        """
        Make a GraphQL request with exponential backoff retry logic.
        Handles 504 Gateway Timeout and other transient errors gracefully.
        """
        variables = variables or {}
        retryable_status_codes = [504, 502, 503, 429]  # Gateway timeout, bad gateway, service unavailable, rate limit
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.url, 
                    json={"query": query, "variables": variables}, 
                    headers=self.headers,
                    timeout=30  # 30 second timeout
                )
                
                # Success case
                if response.status_code == 200:
                    data = response.json()
                    # Check for GraphQL errors
                    if "errors" in data:
                        error_msg = data.get("errors", [{}])[0].get("message", "Unknown GraphQL error")
                        # Some GraphQL errors might be retryable (like rate limits)
                        if "rate limit" in error_msg.lower() and attempt < max_retries - 1:
                            wait_time = (2 ** attempt) + (attempt * 0.5)  # Exponential backoff
                            if on_progress:
                                on_progress(f"⚠️ Rate limit hit, retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        # Non-retryable GraphQL error
                        return response
                    return response
                
                # Retryable HTTP errors
                if response.status_code in retryable_status_codes:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + (attempt * 0.5)  # Exponential backoff: 1s, 2.5s, 5s
                        status_name = {
                            504: "Gateway Timeout",
                            502: "Bad Gateway", 
                            503: "Service Unavailable",
                            429: "Rate Limit"
                        }.get(response.status_code, f"HTTP {response.status_code}")
                        
                        if on_progress:
                            on_progress(f"⚠️ {status_name} - Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        print(f"⚠️ {status_name} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Final attempt failed
                        error_msg = response.text[:200] if response.text else "No error message"
                        if on_progress:
                            on_progress(f"❌ {status_name} after {max_retries} attempts. Please try again later.")
                        print(f"❌ {status_name} after {max_retries} attempts: {error_msg}")
                        return response
                
                # Non-retryable errors (4xx except 429)
                return response
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + (attempt * 0.5)
                    if on_progress:
                        on_progress(f"⚠️ Request timeout - Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    print(f"⚠️ Request timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    if on_progress:
                        on_progress(f"❌ Request timeout after {max_retries} attempts. Please try again later.")
                    print(f"❌ Request timeout after {max_retries} attempts")
                    return None
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + (attempt * 0.5)
                    if on_progress:
                        on_progress(f"⚠️ Network error - Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    print(f"⚠️ Network error (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    if on_progress:
                        on_progress(f"❌ Network error after {max_retries} attempts: {str(e)}")
                    print(f"❌ Network error after {max_retries} attempts: {e}")
                    return None
        
        return None
    
    def get_current_user(self) -> Optional[str]:
        """Get the current authenticated user's login"""
        if self._current_user:
            return self._current_user
        
        query = """
        query {
          viewer {
            login
          }
        }
        """
        response = requests.post(self.url, json={"query": query}, headers=self.headers)
        
        if response.status_code != 200:
            print(f"❌ API Error getting current user: {response.status_code}\n{response.text}")
            return None
        
        data = response.json()
        if "errors" in data:
            print("❌ GraphQL Error:", data["errors"])
            return None
        
        try:
            self._current_user = data["data"]["viewer"]["login"]
            return self._current_user
        except KeyError as e:
            print(f"❌ Error parsing user response: {e}")
            return None
    
    def get_user_prs(self, cursor: str = None, states: List[str] = ["OPEN"], on_progress: Callable[[str], None] = None) -> Optional[Dict[str, Any]]:
        """Get all PRs where the current user is involved (author, reviewer, commenter, etc.)"""
        user_login = self.get_current_user()
        if not user_login:
            return None
        
        if on_progress:
            on_progress(f"🔍 Starting PR search for user {user_login}...")
        
        print(f"--- 🔍 Fetching PRs for user {user_login} (cursor={cursor}, states={states}) ---")
        
        # Build comprehensive search query using GitHub's search syntax
        # GitHub search supports: author, commenter, involves, review-requested, reviewed-by
        # We'll use a comprehensive OR query to catch all involvement types
        state_parts = []
        if "OPEN" in states:
            state_parts.append("is:open")
        if "CLOSED" in states:
            state_parts.append("is:closed")
        if "MERGED" in states:
            state_parts.append("is:merged")
        
        # Build search queries - try most reliable first
        # Based on testing, author: query is most reliable, so we'll try that first
        # Then fall back to other strategies if needed
        
        # GraphQL query template
        query = """
        query($search_query: String!, $cursor: String) {
          search(query: $search_query, type: ISSUE, first: 50, after: $cursor) {
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
                updatedAt
                url
                author {
                  login
                  avatarUrl
                }
                repository {
                  name
                  owner {
                    login
                  }
                  url
                }
                reviewThreads(first: 1) {
                  totalCount
                }
                reviews(first: 20) {
                  nodes {
                    author {
                      login
                    }
                    state
                    createdAt
                  }
                }
                reviewRequests(first: 20) {
                  nodes {
                    requestedReviewer {
                      ... on User {
                        login
                      }
                    }
                  }
                }
                comments(first: 1) {
                  totalCount
                }
                isDraft
                additions
                deletions
                changedFiles
              }
            }
          }
        }
        """
        
        # Strategy 1: Try author search first (most reliable based on testing)
        if len(state_parts) == 1:
            primary_query = f"is:pr author:{user_login} {state_parts[0]}"
        else:
            primary_query = f"is:pr author:{user_login}"
        
        if on_progress:
            on_progress(f"🔍 Strategy 1: Trying author search (most reliable)...")
        print(f"🔍 Primary search query: {primary_query}")
        
        # Try primary query (author search) first
        variables = {"search_query": primary_query, "cursor": cursor}
        response = self._make_request_with_retry(query, variables, on_progress=on_progress)
        
        if response is None:
            if on_progress:
                on_progress(f"❌ Failed to fetch PRs after retries. Please try again later.")
            print(f"❌ Failed to fetch PRs after retries")
            return None
        
        if response.status_code != 200:
            if on_progress:
                on_progress(f"❌ API Error: {response.status_code}")
            print(f"❌ API Error: {response.status_code}\n{response.text[:200]}")
            # Continue to fallback strategies
        else:
            try:
                data = response.json()
                if "errors" in data:
                    if on_progress:
                        on_progress(f"⚠️ Primary query failed, trying fallbacks...")
                    print("❌ GraphQL Error:", data["errors"])
                else:
                    search_result = data["data"]["search"]
                    nodes = search_result.get("nodes", [])
                    if nodes:
                        if on_progress:
                            on_progress(f"✅ Found {len(nodes)} PRs with author search")
                        print(f"✅ Found {len(nodes)} PRs with author search")
                        
                        # Process these results
                        return self._process_pr_results(nodes, search_result, user_login, state_parts, on_progress)
            except (KeyError, ValueError) as e:
                print(f"⚠️ Error parsing response: {e}")
                pass
        
        # If primary query failed or returned no results, try fallback strategies
        if on_progress:
            on_progress(f"🔍 Strategy 2: Trying comprehensive search...")
        
        # Strategy 2: Try comprehensive OR query
        base_query = f"is:pr (author:{user_login} OR commenter:{user_login} OR review-requested:{user_login} OR reviewed-by:{user_login})"
        if len(state_parts) == 1:
            combined_query = f"{base_query} {state_parts[0]}"
        elif len(state_parts) > 1:
            combined_query = base_query
        else:
            combined_query = f"{base_query} is:open"
        
        if on_progress:
            on_progress(f"🔍 Executing: {combined_query}")
        print(f"🔍 Fallback search query: {combined_query}")
        
        variables = {"search_query": combined_query, "cursor": cursor}
        response = self._make_request_with_retry(query, variables, on_progress=on_progress)
        
        if response is None:
            if on_progress:
                on_progress(f"❌ Failed to fetch PRs after retries. Trying individual queries...")
            print(f"❌ Failed to fetch PRs after retries, trying fallbacks")
        elif response.status_code != 200:
            if on_progress:
                on_progress(f"❌ API Error: {response.status_code}")
            print(f"❌ API Error: {response.status_code}\n{response.text[:200]}")
        else:
            try:
                data = response.json()
                if "errors" in data:
                    if on_progress:
                        on_progress(f"⚠️ Comprehensive search failed, trying individual queries...")
                    print("❌ GraphQL Error:", data["errors"])
                else:
                    search_result = data["data"]["search"]
                    nodes = search_result.get("nodes", [])
                    if nodes:
                        if on_progress:
                            on_progress(f"✅ Found {len(nodes)} PRs with comprehensive search")
                        print(f"✅ Found {len(nodes)} PRs with comprehensive search")
                        
                        # Process these results
                        return self._process_pr_results(nodes, search_result, user_login, state_parts, on_progress)
            except (KeyError, ValueError) as e:
                print(f"⚠️ Error parsing response: {e}")
                pass
        
        # Strategy 3: Try individual fallback queries
        return self._try_fallback_queries(user_login, state_parts, query, on_progress)
    
    def _process_pr_results(self, nodes: List[Dict[str, Any]], search_result: Dict[str, Any], 
                           user_login: str, state_parts: List[str], 
                           on_progress: Callable[[str], None] = None) -> Optional[Dict[str, Any]]:
        """Process and filter PR results"""
        try:
            
            # Filter by state if we have multiple states requested (client-side filter)
            if len(state_parts) > 1 and nodes:
                if on_progress:
                    on_progress(f"🔍 Filtering by state...")
                filtered_nodes = []
                for pr in nodes:
                    pr_state = pr.get("state", "").upper()
                    if pr_state in [s.replace("is:", "").upper() for s in state_parts]:
                        filtered_nodes.append(pr)
                nodes = filtered_nodes
                if on_progress:
                    on_progress(f"✅ Filtered to {len(nodes)} PRs matching requested states")
                print(f"✅ Filtered to {len(nodes)} PRs matching requested states")
            
            # Enrich each PR with role information
            if on_progress:
                on_progress(f"🔍 Analyzing user roles for {len(nodes)} PRs...")
            for i, pr in enumerate(nodes):
                pr["user_role"] = self._determine_user_role(pr, user_login)
                if on_progress and len(nodes) > 20 and (i + 1) % 20 == 0:
                    on_progress(f"  → Processed {i + 1}/{len(nodes)} PRs...")
            
            # Filter to only show PRs where user is author or contributor
            filtered_nodes = []
            for pr in nodes:
                roles = pr.get("user_role", [])
                # Only include if user is author or contributor (has reviewed)
                if "author" in roles or "contributor" in roles:
                    filtered_nodes.append(pr)
            
            if on_progress:
                on_progress(f"✅ Filtered to {len(filtered_nodes)} PRs where user is author or contributor (from {len(nodes)} total)")
            print(f"✅ Filtered to {len(filtered_nodes)} PRs where user is author or contributor (from {len(nodes)} total)")
            
            # Update search_result with filtered nodes
            search_result["nodes"] = filtered_nodes
            
            # If we got results, return them
            if filtered_nodes:
                if on_progress:
                    on_progress(f"✨ Complete! Returning {len(filtered_nodes)} PRs")
                return search_result
            
            return None
        except Exception as e:
            if on_progress:
                on_progress(f"❌ Error processing results: {e}")
            print(f"❌ Error processing results: {e}")
            return None
    
    def _try_fallback_queries(self, user_login: str, state_parts: List[str], query: str, 
                              on_progress: Callable[[str], None] = None) -> Optional[Dict[str, Any]]:
        """Try individual fallback queries"""
        if on_progress:
            on_progress(f"🔍 Strategy 3: Trying individual fallback queries...")
        
        # Fallback: Try explicit searches for different involvement types
        fallback_queries = []
        
        # Try commenter and review-requested
        if len(state_parts) == 1:
            fallback_queries.append(f"is:pr commenter:{user_login} {state_parts[0]}")
            fallback_queries.append(f"is:pr review-requested:{user_login} {state_parts[0]}")
        else:
            fallback_queries.append(f"is:pr commenter:{user_login}")
            fallback_queries.append(f"is:pr review-requested:{user_login}")
        
        all_fallback_nodes = []
        seen_pr_numbers = set()
        
        for idx, fallback_query in enumerate(fallback_queries):
            if on_progress:
                on_progress(f"🔍 Fallback {idx + 1}/{len(fallback_queries)}: {fallback_query}")
            print(f"🔍 Trying fallback: {fallback_query}")
            variables = {"search_query": fallback_query, "cursor": None}  # Start fresh for fallbacks
            response = self._make_request_with_retry(query, variables, max_retries=2, on_progress=on_progress)
            
            if response and response.status_code == 200:
                try:
                    data = response.json()
                    if "errors" not in data:
                        fallback_result = data["data"]["search"]
                        fallback_nodes = fallback_result.get("nodes", [])
                        if on_progress and fallback_nodes:
                            on_progress(f"  → Found {len(fallback_nodes)} PRs with this query")
                        for pr in fallback_nodes:
                            # Deduplicate by PR number + repo
                            pr_key = f"{pr.get('repository', {}).get('owner', {}).get('login', '')}/{pr.get('repository', {}).get('name', '')}#{pr.get('number', '')}"
                            if pr_key not in seen_pr_numbers:
                                seen_pr_numbers.add(pr_key)
                                all_fallback_nodes.append(pr)
                except (KeyError, ValueError) as e:
                    print(f"⚠️ Error parsing fallback response: {e}")
                    continue
        
        if all_fallback_nodes:
            if on_progress:
                on_progress(f"✅ Found {len(all_fallback_nodes)} total PRs with fallback searches")
            print(f"✅ Found {len(all_fallback_nodes)} total PRs with fallback searches")
            
            # Process these results
            search_result = {
                "nodes": all_fallback_nodes,
                "pageInfo": {"hasNextPage": False, "endCursor": None}
            }
            return self._process_pr_results(all_fallback_nodes, search_result, user_login, state_parts, on_progress)
        
        if on_progress:
            on_progress(f"⚠️ No PRs found with any search strategy for user {user_login}")
        print(f"⚠️ No PRs found with any search strategy for user {user_login}")
        return {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}}
    
    def _determine_user_role(self, pr: Dict[str, Any], user_login: str) -> List[str]:
        """Determine the user's role(s) in this PR"""
        roles = []
        
        # Check if author
        author_login = pr.get("author", {}).get("login", "").lower() if pr.get("author") else ""
        if author_login == user_login.lower():
            roles.append("author")
        
        # Check if reviewer (requested)
        review_requests = pr.get("reviewRequests", {}).get("nodes", [])
        for req in review_requests:
            reviewer = req.get("requestedReviewer", {})
            reviewer_login = reviewer.get("login", "").lower() if reviewer else ""
            if reviewer_login == user_login.lower():
                roles.append("reviewer")
                break
        
        # Check if has reviewed (contributor)
        reviews = pr.get("reviews", {}).get("nodes", [])
        has_reviewed = False
        for review in reviews:
            review_author = review.get("author", {})
            review_author_login = review_author.get("login", "").lower() if review_author else ""
            if review_author_login == user_login.lower():
                has_reviewed = True
                if "contributor" not in roles:
                    roles.append("contributor")
                break
        
        # Check if they commented (via comments totalCount and if they're in search results)
        # If they appear in search results but aren't author/reviewer/contributor, likely a commenter
        comments_count = pr.get("comments", {}).get("totalCount", 0)
        if comments_count > 0 and not roles:
            roles.append("commenter")
        elif comments_count > 0 and "commenter" not in roles and not has_reviewed:
            # They might have commented but we can't verify from current data
            # Only add if we're sure they're involved somehow
            pass
        
        return roles if roles else ["viewer"]

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
                reviewThreads(first: 1) {
                  totalCount
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

    def search_repositories(self, query: str, first: int = 10, cursor: str = None) -> Optional[Dict[str, Any]]:
        print(f"--- 🔍 Searching Repositories for '{query}' (cursor={cursor}) ---")
        gql_query = """
        query($query: String!, $first: Int!, $cursor: String) {
          search(query: $query, type: REPOSITORY, first: $first, after: $cursor) {
            pageInfo {
              endCursor
              hasNextPage
              hasPreviousPage
              startCursor
            }
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
                isArchived
                pullRequests(states: [OPEN, CLOSED, MERGED]) {
                  totalCount
                }
                issues(states: [OPEN]) {
                  totalCount
                }
              }
            }
          }
        }
        """
        variables = {"query": query, "first": first, "cursor": cursor}
        response = requests.post(self.url, json={"query": gql_query, "variables": variables}, headers=self.headers)
        
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

    def search_issues(self, query: str, first: int = 10) -> Optional[List[Dict[str, Any]]]:
        print(f"--- 🔍 Searching Issues for '{query}' ---")
        gql_query = """
        query($query: String!, $first: Int!) {
          search(query: $query, type: ISSUE, first: $first) {
            nodes {
              ... on Issue {
                number
                title
                url
                state
                createdAt
                author { login }
                repository {
                  name
                  owner { login }
                  stargazerCount
                  primaryLanguage { name }
                }
                body
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

    def get_pr_diff_stats(self, owner: str, repo: str, pr_number: int) -> Optional[Dict[str, int]]:
        print(f"--- 🔍 Fetching Diff Stats for PR #{pr_number} in {owner}/{repo} ---")
        query = """
        query($owner: String!, $repo: String!, $pr_number: Int!) {
          repository(owner: $owner, name: $repo) {
            pullRequest(number: $pr_number) {
              additions
              deletions
              changedFiles
            }
          }
        }
        """
        variables = {"owner": owner, "repo": repo, "pr_number": pr_number}
        response = requests.post(self.url, json={"query": query, "variables": variables}, headers=self.headers)
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        try:
            return data["data"]["repository"]["pullRequest"]
        except (KeyError, TypeError):
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

    def get_pr_threads(self, owner: str, repo: str, pr_number: int) -> Optional[Dict[str, Any]]:
        print(f"--- 🔍 Fetching All Comments for PR #{pr_number} in {owner}/{repo} ---")
        
        # Fetch PR basic info first
        basic_query = """
        query($owner: String!, $repo: String!, $pr_number: Int!) {
          repository(owner: $owner, name: $repo) {
            pullRequest(number: $pr_number) {
              title
              body
              headRefName
              headRefOid
              author { login }
              createdAt
            }
          }
        }
        """
        variables = {"owner": owner, "repo": repo, "pr_number": pr_number}
        response = requests.post(self.url, json={"query": basic_query, "variables": variables}, headers=self.headers)
        
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
            
            # Fetch all review threads with pagination
            all_review_threads = []
            cursor = None
            has_next_page = True
            
            while has_next_page:
                threads_query = """
                query($owner: String!, $repo: String!, $pr_number: Int!, $cursor: String) {
                  repository(owner: $owner, name: $repo) {
                    pullRequest(number: $pr_number) {
                      reviewThreads(first: 100, after: $cursor) {
                        pageInfo {
                          hasNextPage
                          endCursor
                        }
                        nodes {
                          id
                          isResolved
                          isOutdated
                          path
                          comments(first: 100) { 
                            nodes {
                              id
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
                variables = {"owner": owner, "repo": repo, "pr_number": pr_number, "cursor": cursor}
                response = requests.post(self.url, json={"query": threads_query, "variables": variables}, headers=self.headers)
                
                if response.status_code != 200:
                    print(f"⚠️ Error fetching review threads: {response.status_code}")
                    break
                
                data = response.json()
                if "errors" in data:
                    print(f"⚠️ GraphQL Error fetching threads: {data['errors']}")
                    break
                
                threads_data = data["data"]["repository"]["pullRequest"]["reviewThreads"]
                all_review_threads.extend(threads_data.get("nodes", []))
                
                page_info = threads_data.get("pageInfo", {})
                has_next_page = page_info.get("hasNextPage", False)
                cursor = page_info.get("endCursor")
                
                if has_next_page:
                    print(f"📄 Fetched {len(all_review_threads)} review threads so far, fetching more...")
            
            print(f"✅ Fetched {len(all_review_threads)} total review threads")
            
            # Fetch all general comments with pagination
            all_general_comments = []
            cursor = None
            has_next_page = True
            
            while has_next_page:
                comments_query = """
                query($owner: String!, $repo: String!, $pr_number: Int!, $cursor: String) {
                  repository(owner: $owner, name: $repo) {
                    pullRequest(number: $pr_number) {
                      comments(first: 100, after: $cursor) {
                        pageInfo {
                          hasNextPage
                          endCursor
                        }
                        nodes {
                          id
                          author { login }
                          body
                          createdAt
                          isMinimized
                        }
                      }
                    }
                  }
                }
                """
                variables = {"owner": owner, "repo": repo, "pr_number": pr_number, "cursor": cursor}
                response = requests.post(self.url, json={"query": comments_query, "variables": variables}, headers=self.headers)
                
                if response.status_code != 200:
                    print(f"⚠️ Error fetching general comments: {response.status_code}")
                    break
                
                data = response.json()
                if "errors" in data:
                    print(f"⚠️ GraphQL Error fetching comments: {data['errors']}")
                    break
                
                comments_data = data["data"]["repository"]["pullRequest"]["comments"]
                all_general_comments.extend(comments_data.get("nodes", []))
                
                page_info = comments_data.get("pageInfo", {})
                has_next_page = page_info.get("hasNextPage", False)
                cursor = page_info.get("endCursor")
                
                if has_next_page:
                    print(f"📄 Fetched {len(all_general_comments)} general comments so far, fetching more...")
            
            print(f"✅ Fetched {len(all_general_comments)} total general comments")
            
            # Fetch all reviews with pagination
            all_reviews = []
            cursor = None
            has_next_page = True
            
            while has_next_page:
                reviews_query = """
                query($owner: String!, $repo: String!, $pr_number: Int!, $cursor: String) {
                  repository(owner: $owner, name: $repo) {
                    pullRequest(number: $pr_number) {
                      reviews(first: 100, after: $cursor) {
                        pageInfo {
                          hasNextPage
                          endCursor
                        }
                        nodes {
                          id
                          author { login }
                          body
                          state
                          createdAt
                          comments(first: 100) {
                            nodes {
                              id
                              author { login }
                              body
                              createdAt
                              path
                              diffHunk
                            }
                          }
                        }
                      }
                    }
                  }
                }
                """
                variables = {"owner": owner, "repo": repo, "pr_number": pr_number, "cursor": cursor}
                response = requests.post(self.url, json={"query": reviews_query, "variables": variables}, headers=self.headers)
                
                if response.status_code != 200:
                    print(f"⚠️ Error fetching reviews: {response.status_code}")
                    break
                
                data = response.json()
                if "errors" in data:
                    print(f"⚠️ GraphQL Error fetching reviews: {data['errors']}")
                    break
                
                reviews_data = data["data"]["repository"]["pullRequest"]["reviews"]
                all_reviews.extend(reviews_data.get("nodes", []))
                
                page_info = reviews_data.get("pageInfo", {})
                has_next_page = page_info.get("hasNextPage", False)
                cursor = page_info.get("endCursor")
                
                if has_next_page:
                    print(f"📄 Fetched {len(all_reviews)} reviews so far, fetching more...")
            
            print(f"✅ Fetched {len(all_reviews)} total reviews")
            
            # Parse all comment types
            all_items = []
            
            # 1. Review threads (code review comments)
            all_items.extend(self._parse_threads(all_review_threads))
            
            # 2. General PR comments (not on code)
            all_items.extend(self._parse_general_comments(all_general_comments))
            
            # 3. PR body/description (if it has meaningful content)
            if pr_data.get("body") and pr_data.get("body").strip():
                all_items.append(self._parse_pr_body(pr_data))
            
            # 4. Review comments (from review submissions)
            all_items.extend(self._parse_reviews(all_reviews))
            
            print(f"✅ Total items parsed: {len(all_items)} (Review threads: {len(all_review_threads)}, General comments: {len(all_general_comments)}, Reviews: {len(all_reviews)})")
            
            return {
                "items": all_items,
                "headRefName": pr_data["headRefName"],
                "headRefOid": pr_data.get("headRefOid", ""),
                "title": pr_data.get("title", ""),
                "author": pr_data.get("author", {}).get("login", "") if pr_data.get("author") else ""
            }
        except KeyError as e:
            print(f"❌ Error parsing response: {e}")
            import traceback
            traceback.print_exc()
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
    
    def get_pr_diffs(self, owner: str, repo: str, pr_number: int) -> Optional[List[Dict[str, Any]]]:
        """Fetch PR file diffs using REST API"""
        print(f"--- 🔍 Fetching Diffs for PR #{pr_number} ---")
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}\n{response.text}")
            return None
        
        files = response.json()
        result = []
        for file in files:
            result.append({
                "filename": file.get("filename", ""),
                "status": file.get("status", ""),  # added, removed, modified, renamed
                "additions": file.get("additions", 0),
                "deletions": file.get("deletions", 0),
                "changes": file.get("changes", 0),
                "patch": file.get("patch", ""),  # The actual diff content
                "previous_filename": file.get("previous_filename")
            })
        
        return result

    def _parse_threads(self, threads):
        """Parse review threads (code review comments on specific lines)"""
        items = []
        skipped_count = 0
        for thread in threads:
            # Check if thread has comments - handle both None and empty list cases
            comments = thread.get("comments", {})
            comment_nodes = comments.get("nodes", []) if comments else []
            
            if not comment_nodes:
                skipped_count += 1
                continue

            thread_data = {
                "id": thread["id"],
                "type": "review_thread",
                "file_path": thread.get("path"),  # path can be None for general threads
                "status": "RESOLVED" if thread.get("isResolved", False) else "UNRESOLVED",
                "is_outdated": thread.get("isOutdated", False),
                "code_snippet": "", 
                "conversation": []
            }

            # Extract diff hunk
            for c in comment_nodes:
                if c.get("diffHunk"):
                    thread_data["code_snippet"] = c["diffHunk"]
                    break

            # Extract conversation
            for c in comment_nodes:
                # Skip empty comments
                if not c.get("body") or not c.get("body").strip():
                    continue
                    
                msg = {
                    "id": c.get("id", ""),
                    "author": c.get("author", {}).get("login", "Unknown") if c.get("author") else "Unknown",
                    "timestamp": c.get("createdAt", ""),
                    "text": c.get("body", "")
                }
                thread_data["conversation"].append(msg)
            
            # Only add thread if it has at least one valid comment
            if thread_data["conversation"]:
                items.append(thread_data)
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"⚠️ Skipped {skipped_count} threads with no valid comments")
        
        return items

    def _parse_general_comments(self, comments):
        """Parse general PR comments (not tied to specific code lines)"""
        items = []
        for comment in comments:
            # Skip minimized comments
            if comment.get("isMinimized"):
                continue
                
            comment_data = {
                "id": comment["id"],
                "type": "general_comment",
                "file_path": None,
                "status": "ACTIVE",  # General comments don't have resolved status
                "is_outdated": False,
                "code_snippet": "",
                "conversation": [{
                    "id": comment["id"],
                    "author": comment["author"]["login"] if comment["author"] else "Unknown",
                    "timestamp": comment["createdAt"],
                    "text": comment["body"]
                }]
            }
            items.append(comment_data)
        
        return items

    def _parse_pr_body(self, pr_data):
        """Parse PR body/description as a commentable item"""
        return {
            "id": f"pr-body-{pr_data.get('title', '')}",
            "type": "pr_description",
            "file_path": None,
            "status": "ACTIVE",
            "is_outdated": False,
            "code_snippet": "",
            "conversation": [{
                "id": "pr-body",
                "author": pr_data.get("author", {}).get("login", "Unknown") if pr_data.get("author") else "Unknown",
                "timestamp": pr_data.get("createdAt", ""),
                "text": pr_data.get("body", "")
            }]
        }

    def _parse_reviews(self, reviews):
        """Parse review submissions (approve/request changes/comment reviews)"""
        items = []
        for review in reviews:
            # Skip empty reviews
            if not review.get("body") and not review.get("comments", {}).get("nodes"):
                continue
            
            # Create item for review body if present
            if review.get("body"):
                review_data = {
                    "id": review["id"],
                    "type": "review",
                    "file_path": None,
                    "status": review.get("state", "COMMENTED").upper(),
                    "is_outdated": False,
                    "code_snippet": "",
                    "conversation": [{
                        "id": review["id"],
                        "author": review["author"]["login"] if review["author"] else "Unknown",
                        "timestamp": review["createdAt"],
                        "text": review["body"]
                    }]
                }
                items.append(review_data)
            
            # Create items for review comments (inline comments in reviews)
            review_comments = review.get("comments", {}).get("nodes", [])
            for comment in review_comments:
                comment_data = {
                    "id": comment["id"],
                    "type": "review_comment",
                    "file_path": comment.get("path"),
                    "status": review.get("state", "COMMENTED").upper(),
                    "is_outdated": False,
                    "code_snippet": comment.get("diffHunk", ""),
                    "conversation": [{
                        "id": comment["id"],
                        "author": comment["author"]["login"] if comment["author"] else "Unknown",
                        "timestamp": comment["createdAt"],
                        "text": comment["body"]
                    }]
                }
                items.append(comment_data)
        
        return items

class SuggestedContext(BaseModel):
    reasoning: str = Field(..., description="Why these paths were selected based on the PR changes.")
    selected_paths: List[str] = Field(..., description="List of specific file paths or directory paths to index.")

class AIReviewer:
    def __init__(self, api_key: str, base_url: str = None, api_version: str = None, is_azure: bool = False, model: str = "gpt-4o-2024-08-06"):
        self.model_name = model
        if is_azure:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version or "2024-08-01-preview"
            )
        else:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

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
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=SuggestedContext,
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"❌ Error getting context suggestions: {e}")
            return SuggestedContext(reasoning="Error or rate limit. Defaulting to empty context.", selected_paths=[])

    async def analyze_thread(self, thread_data, sandbox_dir: str, custom_prompt: str = None, retriever: VectorStore = None):
        comment_type = thread_data.get('type', 'review_thread')
        
        # Adjust system prompt based on comment type
        if comment_type == 'general_comment' or comment_type == 'review':
            default_system_prompt = (
                "You are an expert Staff Software Engineer analyzing a general PR comment or review. "
                "Your goal is to provide a deep analysis of the discussion and its implications.\n\n"
                "DO NOT just summarize what people said. You must:\n"
                "1. Analyze the Discussion: What technical concerns or questions are being raised?\n"
                "2. Evaluate the Arguments: Assess the technical merit of each point using first principles.\n"
                "3. Identify Action Items: What needs to be addressed based on this conversation?\n"
                "4. Propose Solutions: If issues are raised, what are viable solutions?\n"
                "5. Show Your Work: Fill the 'thinking_process' field with your step-by-step investigation.\n"
            )
        else:
            default_system_prompt = (
                "You are an expert Staff Software Engineer acting as a technical mediator and auditor. "
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
        
        # --- 1. Get Full Content of Target File (if applicable) ---
        target_file_content = ""
        file_path = thread_data.get('file_path')
        if file_path:
            full_file_path = os.path.join(sandbox_dir, file_path)
            try:
                if os.path.exists(full_file_path):
                    with open(full_file_path, "r", encoding="utf-8") as f:
                        target_file_content = f.read()
                else:
                    target_file_content = "(File not found in sandbox - possibly deleted or renamed in this PR)"
            except Exception as e:
                target_file_content = f"(Error reading file: {e})"
        else:
            target_file_content = "(No specific file associated with this comment)"

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
            # Smart Query: Combine file path (if available), diff, and the last comment
            last_comment = thread_data['conversation'][-1]['text'] if thread_data['conversation'] else ""
            file_path = thread_data.get('file_path', '')
            code_snippet = thread_data.get('code_snippet', '')
            
            if file_path:
                query_text = f"File: {file_path}\nDiff: {code_snippet}\nContext: {last_comment}"
            else:
                query_text = f"PR Discussion: {last_comment}"
            
            # Search
            relevant_chunks = retriever.search(query_text, k=5) # Increased k for better coverage
            if relevant_chunks:
                rag_context = "\n--- 📚 RELATED CODE (Dependencies/Usage) ---\n"
                for i, chunk in enumerate(relevant_chunks):
                    meta = chunk['metadata']
                    
                    # Optimization: Skip chunks that are just parts of the target file we already read fully
                    if file_path and meta['file_path'] == file_path:
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

        file_path_display = thread_data.get('file_path', 'General Comment (no file)')
        code_snippet_display = thread_data.get('code_snippet', '')
        if not code_snippet_display:
            code_snippet_display = '(No code snippet - this is a general comment)'
        
        user_content = f"""
        COMMENT TYPE: {comment_type}
        STATUS: {thread_data['status']}
        IS_OUTDATED: {thread_data.get('is_outdated', False)}
        FILE: {file_path_display}
        
        {guidelines_content}
        
        --- 📄 TARGET FILE CONTENT (Current PR Head) ---
        {target_file_content}
        
        {rag_context}
        
        --- 📝 CODE SNIPPET (Original Diff Context) ---
        {code_snippet_display}
        
        --- 💬 CONVERSATION HISTORY ---
        {json.dumps(thread_data['conversation'], indent=2)}
        
        INSTRUCTIONS:
        1. If this is a general comment (not tied to code), focus on the discussion points and action items rather than code analysis.
        2. If IS_OUTDATED is true: The code at the thread location has changed since the comments. You MUST verify if the 'TARGET FILE CONTENT' (which is the latest version) actually implements the requested changes. Do NOT assume it's fixed just because it's outdated.
        3. If STATUS is 'RESOLVED': Verify the fix in the current file content is robust (if applicable).
        4. If STATUS is 'UNRESOLVED': Summarize the blocking issue and write the EXACT code change needed to fix it in 'proposed_fix' (if code-related).
        5. For general comments, provide actionable insights and recommendations.
        """

        try:
            completion = await self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format=CodeSuggestion,
            )
            return {
                "id": thread_data.get("id"),
                "file": thread_data.get("file_path", "General Comment"),
                "status": thread_data["status"],
                "type": comment_type,
                "code_snippet": thread_data.get("code_snippet", ""),
                "rag_sources": rag_sources,
                "ai_analysis": completion.choices[0].message.parsed
            }
        except Exception as e:
            print(f"❌ Error analyzing {thread_data['file_path']}: {e}")
            return None

    async def _specialized_analysis(self, focus_area: str, focus_prompt: str, pr_files: List[str], file_contents: str, guidelines_content: str, rag_context: str, sandbox_dir: str, on_progress: Callable[[str], None] = None) -> Optional[SpecializedAnalysis]:
        """Run a specialized analysis on a specific focus area"""
        system_prompt = (
            f"You are an expert {focus_area} specialist performing a focused code review.\n"
            f"{focus_prompt}\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "- Every finding MUST include the exact file path and line number\n"
            "- Be specific and actionable - no vague statements\n"
            "- Score 1-10 based on this specific area only\n"
            "- If no issues found, state that clearly\n"
        )
        
        user_content = f"""
        {guidelines_content}
        
        --- 📝 MODIFIED FILES IN PR ---
        {file_contents}
        
        {rag_context}
        
        Analyze this PR focusing specifically on {focus_area}. Provide:
        1. Specific findings with file paths and line numbers
        2. Critical issues (if any)
        3. Actionable recommendations
        4. A score (1-10) for this area
        5. A brief summary
        """
        
        try:
            completion = await self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format=SpecializedAnalysis,
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"❌ Error in {focus_area} analysis: {e}")
            return None

    async def general_pr_review(self, pr_files: List[str], sandbox_dir: str, custom_prompt: str = None, retriever: VectorStore = None, indexing_paths: List[str] = None, on_progress: Callable[[str], None] = None) -> Optional[Dict[str, Any]]:
        print(f"🚀 Starting General PR Review for {len(pr_files)} files...")
        if on_progress: on_progress("📖 Reading modified files...")
        
        # 1. Read key files (Limit to prevent context overflow)
        # For a general review, we want to look at the actual code in the modified files
        MAX_FILES_TO_READ = 15  # Increased to get more context
        file_contents = []
        
        for file_path in pr_files[:MAX_FILES_TO_READ]:
             full_path = os.path.join(sandbox_dir, file_path)
             if os.path.exists(full_path):
                 try:
                     with open(full_path, "r", encoding="utf-8") as f:
                         content = f.read(10000) # Increased limit per file
                         file_contents.append(f"--- FILE: {file_path} ---\n{content}\n")
                 except Exception:
                     pass
        
        combined_content = "\n".join(file_contents)
        
        # 2. Get Project Guidelines
        guidelines_content = ""
        guidelines_files = ["CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "STYLE.md", "DEVELOPMENT.md", "README.md"]
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
                        content = f.read(3000) 
                        found_guidelines.append(f"--- {g_file} ---\n{content}\n")
                except:
                    pass
        
        if found_guidelines:
            guidelines_content = "\n--- 📜 PROJECT GUIDELINES & STANDARDS ---\n" + "\n".join(found_guidelines)
        
        # 3. RAG: Retrieve Related Codebase Context
        rag_context = ""
        rag_sources = []
        if retriever:
            # Build queries from PR files to find related code
            queries = []
            for pr_file in pr_files[:5]:  # Use first 5 files for query generation
                # Extract key terms from file path (directory structure, module names)
                path_parts = pr_file.split('/')
                if len(path_parts) > 1:
                    # Use directory as context hint
                    queries.append(f"Module: {'/'.join(path_parts[:-1])}")
                queries.append(f"File: {pr_file}")
            
            # Combine queries and search
            combined_query = " ".join(queries[:3])  # Use first 3 queries
            relevant_chunks = retriever.search(combined_query, k=8)  # Get more context chunks
            
            if relevant_chunks:
                rag_context = "\n--- 📚 CODEBASE CONTEXT (Related Dependencies & Usage) ---\n"
                seen_files = set()
                for i, chunk in enumerate(relevant_chunks):
                    meta = chunk['metadata']
                    file_path = meta['file_path']
                    
                    # Skip if it's one of the PR files we already have
                    if file_path in pr_files:
                        continue
                    
                    # Limit to 1 chunk per file to avoid repetition
                    if file_path in seen_files:
                        continue
                    seen_files.add(file_path)
                    
                    rag_sources.append({
                        "file_path": file_path,
                        "line_start": meta['line_start'],
                        "line_end": meta['line_end'],
                        "content": chunk['text']
                    })
                    rag_context += f"\n[Context {len(rag_sources)}] {file_path} (Lines {meta['line_start']}-{meta['line_end']}):\n"
                    rag_context += chunk['text'] + "\n"
                    
                    if len(rag_sources) >= 6:  # Limit to 6 related files
                        break
        
        # 4. Run Parallel Specialized Analyses
        specialized_focuses = [
            {
                "area": "Security",
                "prompt": "Focus on security vulnerabilities: SQL injection, XSS, authentication/authorization flaws, sensitive data exposure, insecure dependencies, OWASP Top 10 issues. Be critical and flag any security concerns."
            },
            {
                "area": "Performance",
                "prompt": "Focus on performance: N+1 queries, inefficient algorithms (O(n^2) loops), missing indexes, memory leaks, blocking operations, unnecessary computations, caching opportunities. Identify bottlenecks."
            },
            {
                "area": "Architecture",
                "prompt": "Focus on architecture: SOLID principles violations, tight coupling, separation of concerns, design patterns, code organization, technical debt, scalability concerns. Evaluate structural quality."
            },
            {
                "area": "Code Quality",
                "prompt": "Focus on code quality: readability, maintainability, naming conventions, code duplication (DRY), complexity, error handling patterns, documentation, consistency with codebase style."
            },
            {
                "area": "Testing",
                "prompt": "Focus on testing: missing test coverage, edge cases not handled (nulls, empty lists, boundary conditions), test quality, integration test needs, mocking strategies, test maintainability."
            },
            {
                "area": "Code Efficiency & Reusability",
                "prompt": "Focus on minimizing lines changed and maximizing code reuse: Analyze if new classes/models can reuse existing code from the codebase. Summarize what each file change accomplishes and its impact on the wider system. Identify opportunities to refactor existing code instead of creating new implementations. Look for duplicate functionality that could be consolidated. Evaluate if the changes could be achieved with fewer lines by leveraging existing patterns, utilities, or abstractions. Provide specific recommendations with file paths and line numbers for code reuse opportunities."
            },
            {
                "area": "Code Quality & Nitpicking",
                "prompt": "Focus on nitpicking and continuous improvement: Look for ANY opportunities to improve code quality, no matter how small. This includes: naming improvements, code style consistency, missing comments/docstrings, magic numbers that should be constants, unused imports, potential simplifications, better variable names, formatting inconsistencies, type hints where missing, error messages that could be clearer, logging that could be more informative, etc. Every suggestion MUST include valid reasoning explaining WHY the improvement matters (readability, maintainability, debugging, performance, etc.). Be thorough and find at least 5-10 improvement opportunities. Format each as: 'In `file.py` line X: [specific improvement]. Reasoning: [why it matters]'"
            }
        ]
        
        if on_progress: on_progress(f"🔬 Launching {len(specialized_focuses)} parallel specialized analyses...")
        
        # Run specialized analyses in parallel with progress tracking
        async def run_with_progress(focus, index):
            if on_progress: on_progress(f"  → Analyzing {focus['area']} ({index+1}/{len(specialized_focuses)})...")
            result = await self._specialized_analysis(
                focus["area"],
                focus["prompt"],
                pr_files,
                combined_content,
                guidelines_content,
                rag_context,
                sandbox_dir,
                on_progress=on_progress
            )
            if result and on_progress:
                on_progress(f"  ✓ {focus['area']} analysis complete (Score: {result.score}/10)")
            return result
        
        specialized_tasks = [
            run_with_progress(focus, i)
            for i, focus in enumerate(specialized_focuses)
        ]
        
        specialized_results = await asyncio.gather(*specialized_tasks, return_exceptions=True)
        
        # Filter out None and exceptions
        valid_specialized = []
        for i, result in enumerate(specialized_results):
            if result and isinstance(result, SpecializedAnalysis):
                valid_specialized.append(result)
            elif isinstance(result, Exception):
                print(f"⚠️ {specialized_focuses[i]['area']} analysis error: {result}")
                if on_progress: on_progress(f"  ⚠️ {specialized_focuses[i]['area']} analysis failed")
        
        if on_progress: on_progress(f"✅ Completed {len(valid_specialized)}/{len(specialized_focuses)} specialized analyses")
        
        # Combine findings from specialized analyses
        all_critical_issues = []
        all_findings = []
        specialized_summaries = []
        
        for spec in valid_specialized:
            all_critical_issues.extend(spec.critical_issues)
            all_findings.extend(spec.findings)
            specialized_summaries.append(f"{spec.focus_area} (Score: {spec.score}/10): {spec.summary}")
        
        # Build enhanced context for main review
        specialized_context = ""
        if specialized_summaries:
            specialized_context = "\n--- 🔬 PARALLEL SPECIALIZED ANALYSES ---\n"
            specialized_context += "\n".join(specialized_summaries)
            specialized_context += "\n\n--- 🔍 KEY FINDINGS FROM SPECIALIZED ANALYSES ---\n"
            if all_critical_issues:
                specialized_context += "CRITICAL ISSUES:\n" + "\n".join(f"- {issue}" for issue in all_critical_issues[:10]) + "\n\n"
            if all_findings:
                specialized_context += "IMPORTANT FINDINGS:\n" + "\n".join(f"- {finding}" for finding in all_findings[:15]) + "\n"
        
        system_prompt = (
            "You are a collaborative Staff Software Engineer performing a constructive code review.\n"
            "Your goal is to validate the PR changes against existing project patterns and standards, while being pragmatic and encouraging.\n\n"
            "CRITICAL: You have access to:\n"
            "1. The actual modified files in the PR\n"
            "2. Related codebase context (dependencies, usage patterns, similar code)\n"
            "3. Project guidelines and standards (CONTRIBUTING.md)\n\n"
            "REVIEW QUALITY STANDARDS:\n"
            "- **Specificity is Mandatory**: Every recommendation MUST include:\n"
            "  * The exact file path and approximate line number (e.g., 'In `tools.py` around line 45')\n"
            "  * A concrete code example showing the current code and suggested improvement\n"
            "  * Context explaining WHY this change matters\n"
            "- **Key Issues Format**: Each issue must be specific with:\n"
            "  * The exact problem (e.g., 'Missing error handling in `async def fetch_data()` at line 123')\n"
            "  * The impact (e.g., 'Unhandled exceptions will crash the service')\n"
            "  * Reference to the specific code location\n"
            "- **Quality Score Justification**: The score must be justified in the detailed_analysis with:\n"
            "  * What the PR does well (specific examples)\n"
            "  * What prevents it from being 10/10 (specific issues)\n"
            "  * How the score reflects the balance between quality and pragmatism\n"
            "- **Detailed Analysis Depth**: Must include:\n"
            "  * Specific code patterns observed (with file references)\n"
            "  * Comparison with existing codebase patterns (cite RAG context when available)\n"
            "  * Technical trade-offs discussed\n"
            "  * Potential edge cases or failure modes identified\n"
            "  * Performance implications if relevant\n"
            "- **Executive Summary**: Must be comprehensive (3-5 sentences) covering:\n"
            "  * What the PR accomplishes (be specific about features/changes)\n"
            "  * Overall code quality assessment\n"
            "  * Key strengths and any notable concerns\n"
            "  * Merge readiness assessment\n\n"
            "GUIDELINES:\n"
            "- **Tone**: Be helpful, not harsh. Focus on 'Quality Score' as a measure of fit, not just perfection. A score of 8/10 is excellent.\n"
            "- **Context is King**: Prioritize consistency with the existing codebase (RAG context) over generic 'best practices'. If the code matches existing patterns (especially for Vector Search/MongoDB), it is correct.\n"
            "- **Vector Search**: Pay special attention to how Vector Search is implemented. Compare against the provided context to ensure alignment with project-specific implementations.\n"
            "- **Contribution Rules**: If `CONTRIBUTING.md` is provided, strictly follow its style and architectural rules.\n"
            "- **Pragmatism**: Differentiate between 'blocking' issues (bugs, security) and 'nitpicks' (style preferences). Do not block for minor issues.\n"
            "- **Code Examples**: When recommending changes, provide BEFORE/AFTER code snippets showing the exact improvement.\n"
            "- **No Generic Advice**: NEVER say 'ensure X is consistent' without pointing to specific instances where it's inconsistent. NEVER say 'add error handling' without specifying WHERE and WHAT errors to handle."
        )
        
        if custom_prompt:
             system_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_prompt}"
        
        # Enhance system prompt with specialized analysis context
        if valid_specialized:
            system_prompt += "\n\nIMPORTANT: You have access to parallel specialized analyses covering Security, Performance, Architecture, Code Quality, Testing, Code Efficiency & Reusability, and Code Quality & Nitpicking. "
            system_prompt += "Integrate their findings into your comprehensive review. Reference specific findings from specialized analyses when relevant."
        
        if on_progress: on_progress("🧠 Synthesizing comprehensive review from all analyses...")
             
        user_content = f"""
        Please review the following Pull Request changes:
        
        {guidelines_content}
        
        --- 📝 MODIFIED FILES IN PR ---
        {combined_content}
        
        {rag_context}
        
        {specialized_context}
        
        ANALYSIS REQUIREMENTS (STRICT):
        
        1. **Executive Summary** (3-5 sentences):
           - Clearly state what this PR introduces or changes (be specific: new features, refactors, bug fixes)
           - Assess overall code quality and architecture fit
           - Highlight key strengths (cite specific examples)
           - Mention any notable concerns (be specific)
           - Conclude with merge readiness assessment
        
        2. **Key Issues Detected**:
           - Each issue MUST be a structured object with:
             * description: "In `path/to/file.py` around line X: [specific issue]. Impact: [what happens]"
             * file_path: The exact file path (e.g., "holly_api/api/routers/sa_sizing/apc/__init__.py")
             * line_number: Approximate line number where the issue occurs
             * severity: "Critical", "Important", or "Medium"
             * suggested_fix: A COMPLETE, copy-pasteable code snippet showing the fix. Include:
               - Function/class context if needed
               - All necessary imports
               - Proper indentation and syntax
               - CRITICAL: Provide PURE CODE ONLY - NO markdown code fences (no ```python or ```)
               - Example: from fastapi import APIRouter\n\n@router.get("/endpoint")\nasync def handler():\n    try:\n        # Fixed code here\n        pass\n    except Exception as e:\n        logger.error(f"Error: {{e}}")\n        raise
             * rationale: Brief explanation of why this fix is needed
           - Only include genuine issues (bugs, security flaws, architectural problems)
           - If no critical issues, return empty list rather than making up minor ones
        
        3. **Recommendations** (MUST be actionable with copy-pasteable code):
           - EVERY recommendation must be a structured object with:
             * description: "[Severity] In `file.py` line X: [issue description]"
             * file_path: The exact file path
             * line_number: Approximate line number
             * severity: "Critical", "Important", or "Nitpick"
             * current_code: The current code snippet that needs improvement (if applicable)
             * suggested_code: A COMPLETE, copy-pasteable code snippet showing the improvement. Include:
               - Full function/class if context is needed
               - All necessary imports
               - Proper indentation and syntax
               - CRITICAL: Provide PURE CODE ONLY - NO markdown code fences (no ```python or ```)
               - Example: from typing import Optional\n\ndef process_data(data: Optional[dict] = None) -> dict:\n    \"\"\"Processes input data and returns normalized output.\"\"\"\n    if data is None:\n        data = {{}}\n    # ... rest of function
             * rationale: Explanation of why this change matters
           - Include at least 2-3 "Nitpick" suggestions for minor improvements (naming, comments, formatting)
           - CRITICAL: All code snippets must be complete and immediately usable - include imports, function signatures, and proper context
        
        4. **File Change Analysis**:
           - Provide a concise (1-2 sentence) description for EACH modified file
           - Focus on WHAT changed and WHY it matters
           - Be specific: "Adds MongoDB connection pooling with max 10 connections" not "Updates database code"
        
        5. **Detailed Technical Analysis** (comprehensive):
           - **Code Patterns**: Identify specific patterns used (async/await, error handling style, etc.) with file references
           - **Architecture Alignment**: Compare against RAG context - does this match existing patterns? Cite specific examples
           - **Code Quality Assessment**: 
             * What's done well (specific examples with file/line refs)
             * What could be improved (specific examples)
             * Trade-offs made (e.g., "Uses synchronous calls for simplicity, acceptable for this use case")
           - **Edge Cases**: Identify potential failure modes or edge cases (be specific)
           - **Performance**: If relevant, discuss performance implications with specific examples
           - **Security**: If applicable, discuss security considerations
           - **Quality Score Justification**: Explicitly explain why the score is X/10:
             * What earns points (specific strengths)
             * What prevents a perfect score (specific issues)
             * How the score reflects pragmatic assessment
        
        6. **Code Quality Score** (1-10):
           - 9-10: Production-ready, excellent patterns, comprehensive tests, no issues
           - 7-8: Good quality, minor improvements possible, ready to merge
           - 5-6: Functional but needs improvements before merge
           - 1-4: Significant issues, not ready for merge
           - Justify the score in detailed_analysis
        
        CRITICAL: 
        - Every claim must be backed by specific code references (file + line)
        - Every issue MUST include a complete, copy-pasteable code snippet showing the fix
        - Every recommendation MUST include a complete, copy-pasteable code snippet showing the improvement
        - Code snippets must be COMPLETE and USABLE - include imports, function signatures, proper indentation
        - Never use vague language like "consider improving X" without showing HOW with actual code
        - If code follows existing patterns well, explicitly praise it with examples
        - Be thorough but pragmatic - don't block on minor style issues
        
        Provide a structured, actionable review that a developer can immediately act upon by copying and pasting the provided code snippets.
        """
        
        try:
            completion = await self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format=GeneralReview,
            )
            
            # Add specialized analyses to the result
            review_result = completion.choices[0].message.parsed
            
            # Merge specialized findings into the main review
            if valid_specialized:
                # Enhance key_issues with critical issues from specialized analyses
                if all_critical_issues:
                    existing_issues = list(review_result.key_issues) if hasattr(review_result, 'key_issues') else []
                    # Convert string issues to Issue objects if needed (backward compatibility)
                    for issue_text in all_critical_issues[:5]:  # Limit to top 5
                        # Check if it's already an Issue object
                        if isinstance(issue_text, Issue):
                            if issue_text not in existing_issues:
                                existing_issues.append(issue_text)
                        else:
                            # Convert string to Issue object
                            # Try to extract file path from the issue text
                            file_match = re.search(r'`([^`]+\.py)`', issue_text)
                            file_path = file_match.group(1) if file_match else "unknown"
                            line_match = re.search(r'line\s+(\d+)', issue_text)
                            line_num = int(line_match.group(1)) if line_match else None
                            
                            new_issue = Issue(
                                description=issue_text,
                                file_path=file_path,
                                line_number=line_num,
                                severity="Critical",
                                suggested_fix="# TODO: Review and implement fix based on specialized analysis",
                                rationale="Identified by specialized analysis"
                            )
                            # Check if description already exists
                            if not any(iss.description == issue_text for iss in existing_issues):
                                existing_issues.append(new_issue)
                    review_result.key_issues = existing_issues
                
                # Enhance recommendations with specialized recommendations
                existing_recs = list(review_result.recommendations) if hasattr(review_result, 'recommendations') else []
                for spec in valid_specialized:
                    for rec_text in spec.recommendations[:2]:  # Top 2 per area
                        rec_with_tag = f"[{spec.focus_area}] {rec_text}"
                        # Check if it's already a Recommendation object
                        if isinstance(rec_text, Recommendation):
                            if rec_text not in existing_recs:
                                existing_recs.append(rec_text)
                        else:
                            # Convert string to Recommendation object
                            file_match = re.search(r'`([^`]+\.py)`', rec_text)
                            file_path = file_match.group(1) if file_match else "unknown"
                            line_match = re.search(r'line\s+(\d+)', rec_text)
                            line_num = int(line_match.group(1)) if line_match else None
                            severity_match = re.search(r'\[(Critical|Important|Nitpick)\]', rec_text)
                            severity = severity_match.group(1) if severity_match else "Important"
                            
                            new_rec = Recommendation(
                                description=rec_with_tag,
                                file_path=file_path,
                                line_number=line_num,
                                severity=severity,
                                current_code=None,
                                suggested_code="# TODO: Review and implement recommendation based on specialized analysis",
                                rationale="Identified by specialized analysis"
                            )
                            # Check if description already exists
                            if not any(rec.description == rec_with_tag for rec in existing_recs):
                                existing_recs.append(new_rec)
                review_result.recommendations = existing_recs
            
            # Helper function to strip markdown code fences from code snippets
            def strip_code_fences(code: str) -> str:
                """Remove markdown code fences (```python, ```, etc.) from code snippets"""
                if not code or not isinstance(code, str):
                    return code
                # Remove leading/trailing whitespace
                code = code.strip()
                # Remove markdown code fences at start (```python, ```py, ```, etc.)
                # Match: ``` followed by optional language identifier and optional newline
                code = re.sub(r'^```[a-z]*\s*\n?', '', code, flags=re.MULTILINE | re.IGNORECASE)
                # Remove markdown code fences at end (``` with optional whitespace/newlines)
                code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
                return code.strip()
            
            # Convert Pydantic model to dict for serialization (after all modifications)
            # Fast, simple conversion - Pydantic models should serialize easily
            try:
                # Try Pydantic v2 first (most common)
                if hasattr(review_result, 'model_dump'):
                    review_result_dict = review_result.model_dump()
                # Fallback to Pydantic v1
                elif hasattr(review_result, 'dict'):
                    review_result_dict = review_result.dict()
                # Last resort: JSON round-trip (slower but reliable)
                elif hasattr(review_result, 'json'):
                    import json
                    review_result_dict = json.loads(review_result.json())
                else:
                    # Shouldn't happen, but handle gracefully
                    raise ValueError("review_result is not a Pydantic model")
                    
            except Exception as e:
                print(f"❌ Error converting review_result to dict: {e}")
                import traceback
                error_trace = traceback.format_exc()
                print(f"❌ Full traceback:\n{error_trace}")
                if on_progress:
                    on_progress(f"❌ Serialization error: {str(e)}")
                # Re-raise to be caught by outer exception handler
                raise
            
            # Clean up code snippets: strip markdown code fences
            if isinstance(review_result_dict, dict):
                # Clean suggested_fix in key_issues
                if 'key_issues' in review_result_dict:
                    for issue in review_result_dict.get('key_issues', []):
                        if isinstance(issue, dict) and 'suggested_fix' in issue:
                            issue['suggested_fix'] = strip_code_fences(issue['suggested_fix'])
                
                # Clean current_code and suggested_code in recommendations
                if 'recommendations' in review_result_dict:
                    for rec in review_result_dict.get('recommendations', []):
                        if isinstance(rec, dict):
                            if 'current_code' in rec and rec['current_code']:
                                rec['current_code'] = strip_code_fences(rec['current_code'])
                            if 'suggested_code' in rec:
                                rec['suggested_code'] = strip_code_fences(rec['suggested_code'])
            
            # Debug: Log what we're returning
            result_dict = {
                "id": "general-review",
                "file": "General Review",
                "status": "COMPLETED",
                "type": "general",
                "rag_sources": rag_sources,
                "pr_files": pr_files,  # Files changed in the PR
                "indexing_paths": indexing_paths,  # Paths used for vector indexing
                "ai_analysis": review_result_dict,
                "specialized_analyses": valid_specialized
            }
            
            # Validate the result before returning
            if not review_result_dict:
                print(f"⚠️ WARNING: review_result_dict is empty!")
                if on_progress:
                    on_progress("⚠️ Warning: Review result dict is empty")
            
            print(f"✅ Returning result: id={result_dict.get('id')}, ai_analysis keys={list(review_result_dict.keys()) if isinstance(review_result_dict, dict) else 'N/A'}")
            if on_progress:
                on_progress(f"✅ Review completed successfully")
            
            return result_dict
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"❌ Error in General Review: {e}")
            print(f"❌ Full traceback:\n{error_trace}")
            if on_progress:
                on_progress(f"❌ Error: {str(e)}")
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

    async def summarize_analyses(self, thread_results: List[Dict], owner: str, repo: str, pr_number: int, head_ref: str) -> Optional[Dict[str, Any]]:
        print(f"🚀 Summarizing {len(thread_results)} analysis results...")
        
        # Filter successful results
        valid_results = [r for r in thread_results if r]
        if not valid_results:
            return None
            
        # Compile summaries
        summaries = []
        for r in valid_results:
            analysis = r['ai_analysis']
            summaries.append(f"Thread ID: {r['id']}\nFile: {r['file']}\nStatus: {r['status']}\nAI Summary: {analysis.summary}\nSeverity: {analysis.severity}\nProposed Fix: {analysis.proposed_fix}")
        
        combined_summaries = "\n\n".join(summaries)
        
        base_url = f"https://github.com/{owner}/{repo}/blob/{head_ref}"
        
        system_prompt = (
            "You are a Staff Software Engineer / Architect. Your job is to compile a final resolution report based on multiple individual thread analyses.\n"
            "You must identify patterns, blocking issues, and the overall quality of conflict resolution in this PR.\n"
            "CRITICAL: Your output must sound HUMAN. Avoid robotic, stiff, or purely generated-sounding language. Write as if you are a helpful, senior colleague talking directly to the author.\n\n"
            "HYPERLINKING INSTRUCTIONS:\n"
            f"The base URL for files is: {base_url}/<path_to_file>\n"
            "When mentioning a file in the text (Overview, Resolution Audit), ALWAYS wrap it in a Markdown link using the base URL pattern."
        )
        
        user_content = f"""
        Here are the analyses from individual conversation threads in this PR:
        
        {combined_summaries}
        
        INSTRUCTIONS:
        1. Create an Executive Overview of the PR's state.
        2. Identify specific ACTION ITEMS for any UNRESOLVED threads or threads where the fix was deemed inadequate.
        3. Audit the resolution quality: Did the team fix root causes or just symptoms?
        4. Tag the impact areas (e.g., Security, Performance, Style).
        5. Provide a CLEAR recommendation: 'Ready to Merge', 'Request Changes', or 'Blocked'.
        6. Draft specific, HUMAN-sounding comments for the reviewer to post. 
           - **STYLE GUIDE**: Be direct, technical, and authoritative. Avoid "we should consider" or "maybe try".
           - **BAD**: "It might be better to rename this variable for clarity."
           - **GOOD**: "Naming convention mismatch. Rename `sizing_parameters` to `db_update_payload` to match the schema."
           - **BAD**: "There seems to be a missing test case."
           - **GOOD**: "Missing coverage for empty inputs. Add a test case: `assert process([]) == None`."
           - ALWAYS include code snippets or specific variable names in the comment text.
           - Frame suggestions as immediate tasks, not open questions.
        """
        
        try:
            completion = await self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format=AnalysisSummary,
            )
            
            # Post-process Action Items to add URLs
            summary_data = completion.choices[0].message.parsed
            
            # Helper to construct GitHub URL
            def get_gh_url(file_path):
                # Simple blob link to head ref
                return f"https://github.com/{owner}/{repo}/blob/{head_ref}/{file_path}"

            for item in summary_data.action_items:
                item.url = get_gh_url(item.file)

            return {
                "id": "summary-report",
                "file": "Resolution Summary",
                "status": "SUMMARY",
                "ai_summary": summary_data
            }
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return None

    async def answer_followup_question(
        self, 
        question: str, 
        review_result: Dict[str, Any],
        owner: str,
        repo: str,
        pr_number: int,
        pr_files: List[str],
        token: str,
        indexing_paths: List[str] = None
    ) -> str:
        """Answer a follow-up question about a general review with full context"""
        print(f"💬 Answering follow-up question: {question[:50]}...")
        
        # Get the original review analysis
        ai_analysis = review_result.get('ai_analysis')
        if not ai_analysis:
            return "I'm sorry, but I don't have access to the original review data."
        
        # Re-clone the repo to get fresh code context
        # Note: We'll create a vector store here if needed
        # For now, we'll skip vector store re-indexing to keep it simple
        vector_store = None
        
        # Try to get vector store if VOYAGE_API_KEY is available
        voyage_key = os.getenv("VOYAGE_API_KEY")
        if voyage_key:
            vector_store = VectorStore(voyage_key)
        
        async with git_sandbox(owner, repo, token, pr_number=pr_number) as sandbox_dir:
            # Read relevant files for context
            file_contents = []
            for file_path in pr_files[:10]:  # Limit to first 10 files
                full_path = os.path.join(sandbox_dir, file_path)
                if os.path.exists(full_path):
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read(5000)  # Limit per file
                            file_contents.append(f"--- FILE: {file_path} ---\n{content}\n")
                    except Exception:
                        pass
            
            combined_content = "\n".join(file_contents)
            
            # Get RAG context if available (note: vector store may not be indexed, so we skip if empty)
            rag_context = ""
            if vector_store and len(vector_store.documents) > 0:
                try:
                    relevant_chunks = vector_store.search(question, k=5)
                    if relevant_chunks:
                        rag_context = "\n--- 📚 RELEVANT CODEBASE CONTEXT ---\n"
                        for i, chunk in enumerate(relevant_chunks):
                            meta = chunk['metadata']
                            rag_context += f"\n[Context {i+1}] {meta['file_path']} (Lines {meta['line_start']}-{meta['line_end']}):\n"
                            rag_context += chunk['text'] + "\n"
                except Exception as e:
                    print(f"⚠️ Error searching vector store: {e}")
                    # Continue without RAG context
            
            # Build the context from the original review
            review_summary = f"""
ORIGINAL REVIEW SUMMARY:
{ai_analysis.summary}

KEY ISSUES:
{chr(10).join(f"- {issue}" for issue in ai_analysis.key_issues) if ai_analysis.key_issues else "No critical issues detected."}

RECOMMENDATIONS:
{chr(10).join(f"- {rec}" for rec in ai_analysis.recommendations[:5])}  # Limit to first 5

CODE QUALITY SCORE: {ai_analysis.code_quality_score}/10
OVERALL SEVERITY: {ai_analysis.overall_severity}

DETAILED ANALYSIS:
{ai_analysis.detailed_analysis[:2000]}  # Limit to first 2000 chars
"""
            
            system_prompt = (
                "You are a helpful AI assistant answering follow-up questions about a code review.\n"
                "You have access to:\n"
                "1. The complete original review analysis (summary, issues, recommendations, detailed analysis)\n"
                "2. The actual code files from the PR\n"
                "3. Relevant codebase context (if available)\n\n"
                "INSTRUCTIONS:\n"
                "- Answer the user's question directly and concisely\n"
                "- Reference specific parts of the review when relevant\n"
                "- If asked about code, reference specific files and line numbers when possible\n"
                "- If asked about recommendations, explain the rationale behind them\n"
                "- Be helpful, clear, and technical\n"
                "- If you don't have enough information, say so honestly\n"
            )
            
            user_content = f"""
{review_summary}

{rag_context}

--- 📝 PR FILES CONTENT ---
{combined_content}

USER QUESTION: {question}

Please provide a clear, helpful answer to the user's question based on the review and code context above.
"""
            
            try:
                completion = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                answer = completion.choices[0].message.content
                return answer.strip()
                
            except Exception as e:
                print(f"❌ Error answering follow-up question: {e}")
                return f"I encountered an error while processing your question: {str(e)}"
    
    async def chat(self, system_prompt: str, context: str, user_message: str) -> str:
        """Simple chat method for PR conversations"""
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{context}\n\nUser: {user_message}\n\nAssistant:"}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error in chat: {e}")
            return f"I encountered an error: {str(e)}"
