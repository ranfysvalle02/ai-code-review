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
from typing import List, Optional, Dict
from urllib.parse import quote
from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from services import GitHubFetcher, AIReviewer, git_sandbox, VectorStore

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Add custom Jinja2 filter for URL encoding paths
def urlencode_path(path: str) -> str:
    """URL encode a file path, encoding each segment separately"""
    if not path:
        return ""
    # Split by / and encode each segment, then join back
    segments = path.replace('\\', '/').split('/')
    encoded_segments = [quote(segment, safe='') for segment in segments]
    return '/'.join(encoded_segments)

templates.env.filters['urlencode_path'] = urlencode_path

# In-memory store for active progress logs (client_id -> queue)
active_connections: Dict[str, asyncio.Queue] = {}

# In-memory store for analysis jobs (job_id -> dict)
analysis_jobs: Dict[str, Dict] = {}

# In-memory store for completed general reviews (review_id -> review_data)
general_reviews: Dict[str, Dict] = {}

def get_github_fetcher():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="GITHUB_TOKEN not configured")
    return GitHubFetcher(token)

def get_ai_reviewer():
    # Check for Azure OpenAI Configuration
    azure_key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    azure_url = os.getenv("AZURE_OPENAI_URL")
    
    if azure_key and azure_url:
        deployment = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")
        print(f"🔹 Using Azure OpenAI Service (Deployment: {deployment})")
        return AIReviewer(
            api_key=azure_key,
            base_url=azure_url,
            is_azure=True,
            model=deployment, # Azure needs deployment name
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )

    # Fallback to Standard OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY or AZURE_OPENAI_KEY not configured")
    
    print("🔹 Using Standard OpenAI Service")
    return AIReviewer(api_key)

def get_vector_store():
    voyage_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_key:
        print("⚠️ VOYAGE_API_KEY not set. Semantic search will be disabled.")
        return None
    return VectorStore(voyage_key)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    cursor: Optional[str] = None,
    state: str = "OPEN",
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    """HUD Dashboard showing all PRs the user is involved in"""
    # Get current user for display
    current_user = fetcher.get_current_user()
    
    # Return empty initial state - JavaScript will load PRs via SSE
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "prs": [],
        "page_info": {},
        "current_cursor": cursor or "",
        "current_state": state or "OPEN",
        "current_user": current_user or ""
    })

@app.get("/dashboard/stream")
async def dashboard_stream(
    cursor: Optional[str] = None,
    state: str = "OPEN",
    repo: Optional[str] = None,
    role: Optional[str] = None,
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    """Stream PR fetching with progress updates"""
    async def event_generator():
        queue = asyncio.Queue()
        
        def on_progress(msg: str):
            queue.put_nowait({"type": "progress", "msg": msg})
        
        async def worker():
            try:
                # Normalize state to uppercase and validate
                state_upper = state.upper() if state else "OPEN"
                if state_upper not in ["OPEN", "CLOSED", "MERGED"]:
                    state_upper = "OPEN"
                states = [state_upper]
                
                data = fetcher.get_user_prs(cursor=cursor, states=states, on_progress=on_progress)
                
                if data is None:
                    queue.put_nowait({"type": "error", "msg": "Could not fetch your PRs. The request may have timed out or GitHub API is temporarily unavailable. Please try again in a moment."})
                    return
                
                prs = data.get("nodes", [])
                page_info = data.get("pageInfo", {})
                
                # Apply additional filters if provided
                if repo:
                    repo_owner, repo_name = repo.split("/") if "/" in repo else (None, repo)
                    filtered_prs = []
                    for pr in prs:
                        pr_repo_owner = pr.get("repository", {}).get("owner", {}).get("login", "")
                        pr_repo_name = pr.get("repository", {}).get("name", "")
                        if repo_owner:
                            if pr_repo_owner.lower() == repo_owner.lower() and pr_repo_name.lower() == repo_name.lower():
                                filtered_prs.append(pr)
                        else:
                            if pr_repo_name.lower() == repo_name.lower():
                                filtered_prs.append(pr)
                    prs = filtered_prs
                    queue.put_nowait({"type": "progress", "msg": f"🔍 Filtered to {len(prs)} PRs in repository: {repo}"})
                
                if role:
                    filtered_prs = [pr for pr in prs if role.lower() in [r.lower() for r in pr.get("user_role", [])]]
                    prs = filtered_prs
                    queue.put_nowait({"type": "progress", "msg": f"🔍 Filtered to {len(prs)} PRs with role: {role}"})
                
                queue.put_nowait({"type": "result", "prs": prs, "page_info": page_info})
                queue.put_nowait({"type": "done"})
            except Exception as e:
                queue.put_nowait({"type": "error", "msg": str(e)})
        
        # Start worker
        asyncio.create_task(worker())
        
        # Stream events
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                if event["type"] == "done":
                    yield f"data: {json.dumps(event)}\n\n"
                    break
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                yield f": keepalive\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'msg': str(e)})}\n\n"
                break
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    cursor: Optional[str] = None,
    sort: str = "updated-desc",
    language: Optional[str] = None,
    min_stars: int = 50,
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    # Default to environment variable keywords or fallback to "mongodb"
    default_keywords = os.getenv("DEFAULT_SEARCH_TERMS", "mongodb")
    
    # Build query
    query = f"{default_keywords}"
    
    if min_stars:
        query += f" stars:>={min_stars}"
    
    if language and language.strip():
        query += f" language:{language}"
        
    query += " archived:false"
    
    # Handle sort
    # map UI sort values to GitHub qualifiers if needed, or just pass through
    # Expected values: updated-desc, stars-desc, fork-desc
    if sort:
        query += f" sort:{sort}"
    
    # Fetch data
    # We fetch a larger batch if filtering locally, but if we trust query, we can just fetch 9?
    # User wanted "pagination", so we rely on API pagination.
    # But we still have the local filtering for CJK.
    # To support pagination with local filtering is complex because we might filter out all items in a page.
    # For now, let's trust the query + simple filtering.
    
    search_data = fetcher.search_repositories(query, first=12, cursor=cursor)
    
    trending_repos = []
    page_info = {}
    
    if search_data:
        raw_repos = search_data.get("nodes", [])
        page_info = search_data.get("pageInfo", {})
        
        # Apply local content filter
        def is_mostly_latin(text: str) -> bool:
            if not text:
                return True
            cjk_count = 0
            total_count = 0
            for char in text:
                if char.isspace(): continue
                total_count += 1
                code = ord(char)
                if 0x4E00 <= code <= 0x9FFF:
                    cjk_count += 1
            if total_count == 0: return True
            if (cjk_count / total_count) > 0.1:
                return False
            return True

        for repo in raw_repos:
            desc = repo.get('description', '') or ''
            if is_mostly_latin(desc):
                trending_repos.append(repo)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "trending_repos": trending_repos,
        "search_topic": default_keywords,
        "page_info": page_info,
        "current_sort": sort,
        "current_min_stars": min_stars,
        "current_language": language
    })

@app.post("/repos")
async def search_repos(
    owner: str = Form(None),
    repo: str = Form(None),
    search_term: Optional[str] = Form(None)
):
    if not owner or not repo:
         return RedirectResponse(url="/search", status_code=303)
         
    url = f"/repos/{owner}/{repo}"
    if search_term and search_term.strip():
        url += f"?search={search_term.strip()}"
    return RedirectResponse(url=url, status_code=303)

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse("repo_search.html", {"request": request})

@app.post("/search/issues", response_class=HTMLResponse)
async def search_issues(
    request: Request,
    query: str = Form(...),
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    results = fetcher.search_issues(query, first=20)
    return templates.TemplateResponse("repo_search.html", {
        "request": request,
        "issue_results": results,
        "query": query,
        "tab": "issues"
    })

@app.post("/search", response_class=HTMLResponse)
async def perform_search(
    request: Request,
    query: str = Form(...),
    language: Optional[str] = Form(None),
    min_stars: Optional[str] = Form(None),
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    # build query string
    full_query = query
    if language and language.strip():
        full_query += f" language:{language.strip()}"
    
    if min_stars and min_stars.strip():
        try:
            # Validate it's an integer
            stars_val = int(min_stars.strip())
            full_query += f" stars:>={stars_val}"
        except ValueError:
            pass # Ignore invalid input
    
    search_data = fetcher.search_repositories(full_query)
    results = []
    if search_data:
        results = search_data.get("nodes", [])
        
    return templates.TemplateResponse("repo_search.html", {
        "request": request,
        "results": results,
        "query": query,
        "language": language,
        "min_stars": min_stars
    })

@app.get("/repos/{owner}/{repo}", response_class=HTMLResponse)
async def repo_dashboard(
    request: Request,
    owner: str,
    repo: str,
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    # Fetch basic repo info for the header (optional but nice)
    # We can reuse search_repositories for a single repo or get_repo_structure
    # For now, let's just pass owner/repo. If we want details, we can add a lightweight fetch.
    # Let's do a quick search to get star count/desc if possible, or just render the shell.
    
    # Try to get details
    repo_info = None
    try:
        results = fetcher.search_repositories(f"repo:{owner}/{repo}", first=1)
        if results and results.get("nodes"):
            repo_info = results["nodes"][0]
    except:
        pass

    return templates.TemplateResponse("repo_dashboard.html", {
        "request": request,
        "owner": owner,
        "repo": repo,
        "repo_info": repo_info
    })

@app.get("/repos/{owner}/{repo}/prs", response_class=HTMLResponse)
async def list_prs(
    request: Request,
    owner: str,
    repo: str,
    cursor: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = "threads_desc",
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    # If search term is present, use search API (defaults to OPEN state as per requirement)
    # Search API sorting is limited, so we might not be able to sort by thread count easily without fetching all
    if search:
        data = fetcher.search_prs(owner, repo, search, cursor=cursor)
    else:
        # Fetch 100 most recent PRs of mixed states
        data = fetcher.list_prs(owner, repo, states=["OPEN", "CLOSED", "MERGED"], cursor=cursor)
    
    if data is None:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": "Could not fetch PRs. Check owner/repo and token."
        })
    
    prs = data.get("nodes", [])
    
    # Sort by thread count if requested (only works well within the fetched page for cursor-based pagination, 
    # but good enough for 100 items)
    if sort_by == "threads_desc":
        # Sort by total threads descending
        def get_thread_count(pr):
            if "reviewThreads" in pr:
                return pr["reviewThreads"].get("totalCount", 0)
            return 0
        prs.sort(key=get_thread_count, reverse=True)
    
    page_info = data.get("pageInfo", {})
    
    return templates.TemplateResponse("prs.html", {
        "request": request, 
        "owner": owner, 
        "repo": repo, 
        "prs": prs,
        "page_info": page_info,
        "current_cursor": cursor,
        "search_term": search,
        "current_sort": sort_by
    })

@app.get("/repos/{owner}/{repo}/prs/{pr_number}", response_class=HTMLResponse)
async def pr_detail(
    request: Request,
    owner: str,
    repo: str,
    pr_number: int,
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    result = fetcher.get_pr_threads(owner, repo, pr_number)
    if result is None:
         return templates.TemplateResponse("prs.html", {
            "request": request, 
            "owner": owner, 
            "repo": repo, 
            "error": "Could not fetch comments."
        })

    return templates.TemplateResponse("pr_detail.html", {
        "request": request,
        "owner": owner,
        "repo": repo,
        "pr_number": pr_number,
        "items": result.get("items", []),
        "pr_title": result.get("title", ""),
        "pr_author": result.get("author", "")
    })

@app.get("/api/prs/{owner}/{repo}/{pr_number}/comments")
async def get_pr_comments(
    owner: str,
    repo: str,
    pr_number: int,
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    """API endpoint to fetch PR comments for preview modal"""
    result = fetcher.get_pr_threads(owner, repo, pr_number)
    if result is None:
        raise HTTPException(status_code=404, detail="Could not fetch PR comments")
    
    return {
        "title": result.get("title", ""),
        "author": result.get("author", ""),
        "items": result.get("items", [])
    }

@app.get("/repos/{owner}/{repo}/prs/{pr_number}/comments", response_class=HTMLResponse)
async def pr_comments_page(
    request: Request,
    owner: str,
    repo: str,
    pr_number: int,
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    """Page showing PR comments, code changes, and AI chat"""
    # Fetch comments
    comments_result = fetcher.get_pr_threads(owner, repo, pr_number)
    if comments_result is None:
        return templates.TemplateResponse("prs.html", {
            "request": request,
            "owner": owner,
            "repo": repo,
            "error": "Could not fetch PR comments."
        })
    
    # Fetch PR diffs
    diffs = fetcher.get_pr_diffs(owner, repo, pr_number)
    
    return templates.TemplateResponse("pr_comments.html", {
        "request": request,
        "owner": owner,
        "repo": repo,
        "pr_number": pr_number,
        "pr_title": comments_result.get("title", ""),
        "pr_author": comments_result.get("author", ""),
        "comments": comments_result.get("items", []),
        "diffs": diffs or []
    })

@app.post("/api/prs/{owner}/{repo}/{pr_number}/chat")
async def pr_chat(
    owner: str,
    repo: str,
    pr_number: int,
    request: Request,
    fetcher: GitHubFetcher = Depends(get_github_fetcher),
    reviewer: AIReviewer = Depends(get_ai_reviewer)
):
    """API endpoint for AI chat about PR"""
    data = await request.json()
    user_message = data.get("message", "")
    
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    # Get PR context
    comments_result = fetcher.get_pr_threads(owner, repo, pr_number)
    diffs = fetcher.get_pr_diffs(owner, repo, pr_number)
    
    # Build context for AI
    pr_title = comments_result.get('title', 'Unknown PR') if comments_result else 'Unknown PR'
    context = f"Pull Request #{pr_number} in {owner}/{repo}\n"
    context += f"Title: {pr_title}\n"
    context += f"Author: {comments_result.get('author', 'Unknown') if comments_result else 'Unknown'}\n\n"
    
    if comments_result and comments_result.get("items"):
        context += f"Comments ({len(comments_result['items'])} total):\n"
        for i, item in enumerate(comments_result["items"][:5], 1):  # Limit to first 5 items
            file_path = item.get("file_path", "General comment")
            status = item.get("status", "")
            context += f"{i}. {file_path} [{status}]\n"
            if item.get("code_snippet"):
                context += f"   Code: {item.get('code_snippet')[:200]}...\n"
            if item.get("conversation"):
                last_msg = item.get("conversation")[-1] if item.get("conversation") else {}
                context += f"   Last comment: {last_msg.get('text', '')[:150]}...\n"
        context += "\n"
    
    if diffs:
        context += f"Changed Files ({len(diffs)}):\n"
        for diff in diffs[:5]:  # Limit to first 5 files
            filename = diff.get('filename', '')
            additions = diff.get('additions', 0)
            deletions = diff.get('deletions', 0)
            status = diff.get('status', '')
            context += f"- {filename} [{status}] (+{additions}, -{deletions})\n"
            if diff.get('patch'):
                patch_preview = diff.get('patch', '')[:300]
                context += f"  Diff preview: {patch_preview}...\n"
    
    # Get AI response
    try:
        system_prompt = (
            f"You are a helpful AI assistant helping review Pull Request #{pr_number} in {owner}/{repo}. "
            "You have access to information about the PR's comments, code changes, and file modifications. "
            "Answer questions clearly and concisely. Reference specific files, comments, or code when relevant. "
            "If asked about code quality, security, or best practices, provide thoughtful analysis."
        )
        response = await reviewer.chat(system_prompt, context, user_message)
        return {"response": response}
    except Exception as e:
        print(f"❌ Error in PR chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting AI response: {str(e)}")

@app.get("/repos/{owner}/{repo}/structure")
async def get_repo_structure(
    owner: str,
    repo: str,
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    # Fetch full file tree for the picker
    files = fetcher.get_file_tree(owner, repo)
    if not files:
        # Fallback to structure if tree fails
        files = fetcher.get_repo_structure(owner, repo)
        
    if files is None:
        raise HTTPException(status_code=404, detail="Could not fetch repo structure")
    return {"directories": files}

@app.get("/repos/{owner}/{repo}/prs/{pr_number}/suggest_context")
async def suggest_context(
    owner: str,
    repo: str,
    pr_number: int,
    fetcher: GitHubFetcher = Depends(get_github_fetcher),
    reviewer: AIReviewer = Depends(get_ai_reviewer)
):
    # 1. Get Repo File Tree (Recursive)
    file_tree = fetcher.get_file_tree(owner, repo)
    if not file_tree:
        # Fallback to simple structure if recursive fails
        file_tree = fetcher.get_repo_structure(owner, repo) or []

    # 2. Get PR Files
    pr_files = fetcher.get_pr_files(owner, repo, pr_number)
    
    # 3. Ask AI
    suggestion = await reviewer.suggest_indexing_paths(file_tree, pr_files)
    
    return suggestion

@app.get("/progress/{client_id}")
async def progress_stream(client_id: str):
    async def event_generator():
        # Yield initial connection message
        yield "data: CONNECTED\n\n"
        
        queue = asyncio.Queue()
        active_connections[client_id] = queue
        try:
            while True:
                msg = await queue.get()
                if msg == "DONE":
                    break
                yield f"data: {msg}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            active_connections.pop(client_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/analyze/job/{job_id}/stream")
async def stream_analysis_job(
    job_id: str,
    fetcher: GitHubFetcher = Depends(get_github_fetcher),
    reviewer: AIReviewer = Depends(get_ai_reviewer)
):
    job_data = analysis_jobs.get(job_id)
    if not job_data:
        # Return a stream that immediately errors
        async def error_generator():
            yield f"data: {json.dumps({'type': 'error', 'msg': 'Job not found'})}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")

    async def event_generator():
        # Create a queue for logs/events
        queue = asyncio.Queue()
        
        # Callback to push logs
        def on_progress(msg):
            # Push to stream
            queue.put_nowait({"type": "log", "msg": msg})

        # The worker task that runs the heavy logic
        async def worker():
            try:
                token = os.getenv("GITHUB_TOKEN")
                
                if job_data.get('review_type') == 'general':
                    # --- GENERAL REVIEW ---
                    on_progress("📋 Step 1/4: Fetching PR modified files...")
                    
                    # GUARDRAIL: Check for large PRs
                    diff_stats = fetcher.get_pr_diff_stats(job_data['owner'], job_data['repo'], job_data['pr_number'])
                    if diff_stats:
                        total_changes = diff_stats.get('additions', 0) + diff_stats.get('deletions', 0)
                        if total_changes > 1000:
                            on_progress(f"⚠️ LARGE PR DETECTED ({total_changes} lines). Human review recommended.")
                            queue.put_nowait({
                                "type": "warning", 
                                "msg": f"This PR has {total_changes} lines of code changed. AI accuracy may degrade. Human-in-the-loop is strictly required."
                            })
                    
                    pr_files = fetcher.get_pr_files(job_data['owner'], job_data['repo'], job_data['pr_number'])
                    
                    if not pr_files:
                        queue.put_nowait({"type": "error", "msg": "No modified files found in this PR."})
                        return
                    
                    # Get PR head commit SHA for file links (more reliable than branch name)
                    pr_data = fetcher.get_pr_threads(job_data['owner'], job_data['repo'], job_data['pr_number'])
                    head_commit_sha = pr_data.get("headRefOid", "") if pr_data else ""
                    head_ref = pr_data.get("headRefName", "main") if pr_data else "main"
                    
                    on_progress(f"✓ Found {len(pr_files)} modified files")

                    async with git_sandbox(job_data['owner'], job_data['repo'], token, pr_number=job_data['pr_number'], on_progress=on_progress) as sandbox_dir:
                        # Setup Vector Store & Indexing for codebase context
                        vector_store = get_vector_store()
                        if vector_store:
                            on_progress("📚 Step 2/4: Indexing codebase for context (this may take 30-60 seconds)...")
                            stats = await vector_store.index_repo(sandbox_dir, include_paths=job_data.get('indexing_paths'), on_progress=on_progress)
                            queue.put_nowait({"type": "indexed", "stats": stats})
                            on_progress(f"✓ Indexed {stats.get('total_files', 0)} files")
                        
                        on_progress("🔬 Step 3/4: Launching parallel specialized analyses (5 AI agents)...")
                        
                        result = await reviewer.general_pr_review(
                            pr_files, 
                            sandbox_dir, 
                            job_data.get('custom_prompt'),
                            retriever=vector_store,
                            indexing_paths=job_data.get('indexing_paths'),
                            on_progress=on_progress
                        )
                        
                        on_progress("📊 Step 4/4: Finalizing review report...")
                        
                        if result:
                            # Store review data for follow-up questions
                            review_id = f"{job_id}-general-review"
                            
                            # Update result id to match review_id for template rendering
                            result['id'] = review_id
                            
                            general_reviews[review_id] = {
                                "review_result": result,
                                "owner": job_data['owner'],
                                "repo": job_data['repo'],
                                "pr_number": job_data['pr_number'],
                                "pr_files": pr_files,
                                "sandbox_dir": sandbox_dir,  # Note: sandbox_dir will be cleaned up, but we can re-clone if needed
                                "vector_store": vector_store,  # Note: vector_store is in-memory, may not persist
                                "indexing_paths": job_data.get('indexing_paths')
                            }
                            
                            # Render HTML partial (now res.id will be the review_id)
                            html_content = templates.get_template("partials/general_result.html").render(
                                res=result,
                                owner=job_data['owner'],
                                repo=job_data['repo'],
                                pr_number=job_data['pr_number'],
                                head_ref=head_ref,
                                head_commit_sha=head_commit_sha
                            )
                            payload = {
                                "id": review_id,
                                "file": result['file'],
                                "status": result['status'],
                                "html": html_content
                            }
                            queue.put_nowait({"type": "result", "payload": payload})
                            on_progress("✅ General Review Complete")
                        else:
                            queue.put_nowait({"type": "error", "msg": "AI Review failed."})

                else:
                    # --- COMMENT/THREAD ANALYSIS ---
                    # 1. Fetch All Comments
                    on_progress("🔍 Fetching all PR comments...")
                    # Update: get_pr_threads now returns a dict {items, headRefName, title, author}
                    pr_data = fetcher.get_pr_threads(job_data['owner'], job_data['repo'], job_data['pr_number'])
                    if not pr_data:
                        queue.put_nowait({"type": "error", "msg": "Could not fetch PR data"})
                        return
                        
                    all_items = pr_data.get("items", [])
                    head_ref = pr_data.get("headRefName", "main")
                    
                    # 2. Filter by selected IDs
                    selected_ids = set(job_data.get('selected_threads', []))
                    items_to_analyze = [item for item in all_items if item.get("id") in selected_ids]
                    
                    if not items_to_analyze:
                        queue.put_nowait({"type": "error", "msg": "No comments/threads selected"})
                        return

                    # 3. Setup Sandbox & Indexing
                    vector_store = get_vector_store()
                    
                    async with git_sandbox(job_data['owner'], job_data['repo'], token, pr_number=job_data['pr_number'], on_progress=on_progress) as sandbox_dir:
                        if vector_store:
                            stats = await vector_store.index_repo(sandbox_dir, include_paths=job_data['indexing_paths'], on_progress=on_progress)
                            queue.put_nowait({"type": "indexed", "stats": stats})
                        
                        on_progress("🚀 Launching AI Agents...")
                        
                        # 4. Run Analysis in Parallel
                        tasks = [
                            reviewer.analyze_thread(item, sandbox_dir, job_data.get('custom_prompt'), retriever=vector_store) 
                            for item in items_to_analyze
                        ]
                        
                        results_list = []
                        for future in asyncio.as_completed(tasks):
                            result = await future
                            if result:
                                results_list.append(result)
                                # Render HTML partial for this result
                                html_content = templates.get_template("partials/thread_result.html").render(res=result)
                                payload = {
                                    "id": result['id'],
                                    "file": result.get('file', 'General Comment'),
                                    "status": result['status'],
                                    "html": html_content
                                }
                                queue.put_nowait({"type": "result", "payload": payload})
                                on_progress(f"✅ Analysis complete for {result.get('file', 'comment')}")
                        
                        # 5. Generate Summary if we have results
                        if results_list:
                            on_progress("📊 Generating final resolution summary...")
                            summary = await reviewer.summarize_analyses(
                                results_list, 
                                job_data['owner'], 
                                job_data['repo'], 
                                job_data['pr_number'],
                                head_ref
                            )
                            if summary:
                                # Add stats
                                summary['resolved_count'] = sum(1 for r in results_list if r['status'] in ['RESOLVED', 'APPROVED'])
                                summary['unresolved_count'] = sum(1 for r in results_list if r['status'] not in ['RESOLVED', 'APPROVED', 'ACTIVE'])
                                summary['total_threads'] = len(results_list)

                                html_content = templates.get_template("partials/analysis_summary.html").render(res=summary)
                                payload = {
                                    "id": summary['id'],
                                    "file": summary['file'],
                                    "status": summary['status'],
                                    "html": html_content
                                }
                                queue.put_nowait({"type": "result", "payload": payload})
                                on_progress("✅ Resolution Summary Generated")
                
                queue.put_nowait({"type": "done"})
                
            except Exception as e:
                print(f"Worker Error: {e}")
                queue.put_nowait({"type": "error", "msg": str(e)})

        # Start worker
        task = asyncio.create_task(worker())

        # Consume queue
        while True:
            # Wait for messages
            data = await queue.get()
            yield f"data: {json.dumps(data)}\n\n"
            
            if data['type'] in ['done', 'error']:
                break
        
        # clean up
        analysis_jobs.pop(job_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_threads(
    request: Request,
    owner: str = Form(...),
    repo: str = Form(...),
    pr_number: int = Form(...),
    review_type: str = Form("threads"),
    selected_threads: List[str] = Form([]),
    indexing_paths: List[str] = Form(None),
    custom_prompt: Optional[str] = Form(None)
):
    import uuid
    job_id = str(uuid.uuid4())
    
    # Store job data
    analysis_jobs[job_id] = {
        "owner": owner,
        "repo": repo,
        "pr_number": pr_number,
        "review_type": review_type,
        "selected_threads": selected_threads,
        "indexing_paths": indexing_paths,
        "custom_prompt": custom_prompt
    }
    
    # Render the dashboard shell immediately
    return templates.TemplateResponse("analysis_dashboard.html", {
        "request": request,
        "job_id": job_id,
        "owner": owner,
        "repo": repo,
        "pr_number": pr_number
    })

@app.post("/analyze/followup")
async def handle_followup_question(
    request: Request,
    reviewer: AIReviewer = Depends(get_ai_reviewer),
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    """Handle follow-up questions about a general review"""
    try:
        data = await request.json()
        review_id = data.get("review_id")
        question = data.get("question")
        
        if not review_id or not question:
            print(f"❌ Missing review_id or question. review_id={review_id}, question={question[:50] if question else None}")
            raise HTTPException(status_code=400, detail="review_id and question are required")
        
        print(f"💬 Follow-up question for review_id: {review_id}")
        print(f"📝 Available reviews: {list(general_reviews.keys())}")
        
        # Get stored review data
        review_data = general_reviews.get(review_id)
        if not review_data:
            print(f"❌ Review not found: {review_id}")
            raise HTTPException(status_code=404, detail=f"Review not found. Available reviews: {list(general_reviews.keys())[:3]}")
        
        # Get the review result
        review_result = review_data["review_result"]
        
        # Re-clone the repo if needed (sandbox was cleaned up)
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise HTTPException(status_code=500, detail="GITHUB_TOKEN not configured")
        
        # Get PR files again
        pr_files = fetcher.get_pr_files(
            review_data["owner"], 
            review_data["repo"], 
            review_data["pr_number"]
        )
        
        # Answer the follow-up question
        answer = await reviewer.answer_followup_question(
            question=question,
            review_result=review_result,
            owner=review_data["owner"],
            repo=review_data["repo"],
            pr_number=review_data["pr_number"],
            pr_files=pr_files,
            token=token,
            indexing_paths=review_data.get("indexing_paths")
        )
        
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Error answering follow-up: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")
