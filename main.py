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
from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from services import GitHubFetcher, AIReviewer, git_sandbox, VectorStore

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# In-memory store for active progress logs (client_id -> queue)
active_connections: Dict[str, asyncio.Queue] = {}

# In-memory store for analysis jobs (job_id -> dict)
analysis_jobs: Dict[str, Dict] = {}

def get_github_fetcher():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="GITHUB_TOKEN not configured")
    return GitHubFetcher(token)

def get_ai_reviewer():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    return AIReviewer(api_key)

def get_vector_store():
    voyage_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_key:
        print("⚠️ VOYAGE_API_KEY not set. Semantic search will be disabled.")
        return None
    return VectorStore(voyage_key)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/repos")
async def search_repos(
    owner: str = Form(...),
    repo: str = Form(...),
    search_term: Optional[str] = Form(None)
):
    url = f"/repos/{owner}/{repo}"
    if search_term and search_term.strip():
        url += f"?search={search_term.strip()}"
    return RedirectResponse(url=url, status_code=303)

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse("repo_search.html", {"request": request})

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
    
    results = fetcher.search_repositories(full_query)
    return templates.TemplateResponse("repo_search.html", {
        "request": request,
        "results": results,
        "query": query,
        "language": language,
        "min_stars": min_stars
    })

@app.get("/repos/{owner}/{repo}", response_class=HTMLResponse)
async def list_prs(
    request: Request,
    owner: str,
    repo: str,
    cursor: Optional[str] = None,
    search: Optional[str] = None,
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    # If search term is present, use search API (defaults to OPEN state as per requirement)
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
    page_info = data.get("pageInfo", {})
    
    return templates.TemplateResponse("prs.html", {
        "request": request, 
        "owner": owner, 
        "repo": repo, 
        "prs": prs,
        "page_info": page_info,
        "current_cursor": cursor,
        "search_term": search
    })

@app.get("/repos/{owner}/{repo}/prs/{pr_number}", response_class=HTMLResponse)
async def pr_detail(
    request: Request,
    owner: str,
    repo: str,
    pr_number: int,
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    threads = fetcher.get_pr_threads(owner, repo, pr_number)
    if threads is None:
         return templates.TemplateResponse("prs.html", {
            "request": request, 
            "owner": owner, 
            "repo": repo, 
            "error": "Could not fetch threads."
        })

    return templates.TemplateResponse("pr_detail.html", {
        "request": request,
        "owner": owner,
        "repo": repo,
        "pr_number": pr_number,
        "threads": threads
    })

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
                # 1. Fetch Threads
                on_progress("🔍 Fetching threads...")
                all_threads = fetcher.get_pr_threads(job_data['owner'], job_data['repo'], job_data['pr_number'])
                
                # 2. Filter
                selected_ids = set(job_data['selected_threads'])
                threads_to_analyze = [t for t in all_threads if t["id"] in selected_ids]
                
                if not threads_to_analyze:
                    queue.put_nowait({"type": "error", "msg": "No threads found"})
                    return

                # 3. Setup Sandbox & Indexing
                token = os.getenv("GITHUB_TOKEN")
                vector_store = get_vector_store()
                
                async with git_sandbox(job_data['owner'], job_data['repo'], token, pr_number=job_data['pr_number'], on_progress=on_progress) as sandbox_dir:
                    if vector_store:
                        stats = await vector_store.index_repo(sandbox_dir, include_paths=job_data['indexing_paths'], on_progress=on_progress)
                        queue.put_nowait({"type": "indexed", "stats": stats})
                    
                    on_progress("🚀 Launching AI Agents...")
                    
                    # 4. Run Analysis in Parallel
                    tasks = [
                        reviewer.analyze_thread(t, sandbox_dir, job_data['custom_prompt'], retriever=vector_store) 
                        for t in threads_to_analyze
                    ]
                    
                    for future in asyncio.as_completed(tasks):
                        result = await future
                        if result:
                            # Render HTML partial for this result
                            html_content = templates.get_template("partials/thread_result.html").render(res=result)
                            payload = {
                                "id": result['id'],
                                "file": result['file'],
                                "status": result['status'],
                                "html": html_content
                            }
                            queue.put_nowait({"type": "result", "payload": payload})
                            on_progress(f"✅ Analysis complete for {result['file']}")
                
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
