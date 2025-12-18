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
import logging
import copy
import re
from typing import List, Optional, Dict
from urllib.parse import quote
from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from services import GitHubFetcher, AIReviewer, git_sandbox, VectorStore
from cache import IntelligentCache

logger = logging.getLogger(__name__)

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

def render_markdown(text: str) -> str:
    """Convert markdown text to HTML with proper formatting"""
    if not text:
        return ""
    
    import html
    
    # First, convert literal \n to actual newlines
    text = text.replace('\\n', '\n')
    
    # Store and replace code blocks first (before any processing)
    code_block_placeholders = []
    def store_code_block(match):
        idx = len(code_block_placeholders)
        code_block_placeholders.append(match.group(0))
        return f'__CODE_BLOCK_{idx}__'
    text = re.sub(r'```[\s\S]*?```', store_code_block, text)
    
    # Process line by line for structure (headers, lists, paragraphs)
    lines = text.split('\n')
    result_lines = []
    in_list = False
    list_type = None
    current_paragraph = []
    
    def process_inline_markdown(text: str) -> str:
        """Process inline markdown (bold, code) in text"""
        # Escape HTML first
        text = html.escape(text)
        # Bold text **text** or __text__
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong class="font-bold text-white">\1</strong>', text)
        text = re.sub(r'__(.*?)__', r'<strong class="font-bold text-white">\1</strong>', text)
        # Inline code `code`
        text = re.sub(r'`([^`]+?)`', r'<code class="bg-gray-800 px-1.5 py-0.5 rounded text-xs font-mono text-green-400">\1</code>', text)
        return text
    
    def flush_paragraph():
        nonlocal current_paragraph
        if current_paragraph:
            para_text = ' '.join(current_paragraph)
            if para_text.strip():
                processed_text = process_inline_markdown(para_text)
                result_lines.append(f'<p class="mb-3">{processed_text}</p>')
            current_paragraph = []
    
    def flush_list():
        nonlocal in_list, list_type
        if in_list:
            tag = '</ol>' if list_type == 'ordered' else '</ul>'
            result_lines.append(tag)
            in_list = False
            list_type = None
    
    for line in lines:
        # Headers
        if re.match(r'^###\s+(.+)$', line):
            flush_paragraph()
            flush_list()
            header_text = re.sub(r'^###\s+', '', line)
            result_lines.append(f'<h3 class="text-lg font-bold text-white mt-4 mb-2">{process_inline_markdown(header_text)}</h3>')
            continue
        elif re.match(r'^##\s+(.+)$', line):
            flush_paragraph()
            flush_list()
            header_text = re.sub(r'^##\s+', '', line)
            result_lines.append(f'<h2 class="text-xl font-bold text-white mt-5 mb-3">{process_inline_markdown(header_text)}</h2>')
            continue
        elif re.match(r'^#\s+(.+)$', line):
            flush_paragraph()
            flush_list()
            header_text = re.sub(r'^#\s+', '', line)
            result_lines.append(f'<h1 class="text-2xl font-bold text-white mt-6 mb-4">{process_inline_markdown(header_text)}</h1>')
            continue
        
        # Lists
        if re.match(r'^\s*[-*]\s+', line):
            flush_paragraph()
            if not in_list or list_type != 'unordered':
                flush_list()
                result_lines.append('<ul class="list-disc list-inside space-y-1 my-2 ml-4">')
                in_list = True
                list_type = 'unordered'
            content = re.sub(r'^\s*[-*]\s+', '', line)
            result_lines.append(f'<li class="mb-1">{process_inline_markdown(content)}</li>')
            continue
        elif re.match(r'^\s*\d+\.\s+', line):
            flush_paragraph()
            if not in_list or list_type != 'ordered':
                flush_list()
                result_lines.append('<ol class="list-decimal list-inside space-y-1 my-2 ml-4">')
                in_list = True
                list_type = 'ordered'
            content = re.sub(r'^\s*\d+\.\s+', '', line)
            result_lines.append(f'<li class="mb-1">{process_inline_markdown(content)}</li>')
            continue
        
        # Empty line
        if not line.strip():
            flush_paragraph()
            flush_list()
            continue
        
        # Regular paragraph line
        flush_list()
        current_paragraph.append(line)
    
    # Flush any remaining content
    flush_paragraph()
    flush_list()
    
    text = '\n'.join(result_lines)
    
    # Restore code blocks and process them
    for idx, code_block in enumerate(code_block_placeholders):
        placeholder = f'__CODE_BLOCK_{idx}__'
        if placeholder in text:
            # Extract language and code
            lang_match = re.match(r'```(\w+)?', code_block)
            language = lang_match.group(1) if lang_match else ''
            code = re.sub(r'```\w*\n?', '', code_block)
            code = re.sub(r'```$', '', code).strip()
            code_escaped = html.escape(code)
            lang_class = f'language-{language}' if language else ''
            code_html = f'<pre class="bg-gray-900 rounded-lg p-4 border border-gray-700 overflow-x-auto my-4"><code class="text-xs font-mono text-gray-300 {lang_class}">{code_escaped}</code></pre>'
            text = text.replace(placeholder, code_html)
    
    return text

templates.env.filters['urlencode_path'] = urlencode_path
templates.env.filters['markdown'] = render_markdown

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

# Initialize intelligent cache (MongoDB if available, in-memory otherwise)
_cache_instance: Optional[IntelligentCache] = None

def get_cache() -> IntelligentCache:
    """Get or create the cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = IntelligentCache()
        logger.info(f"✅ Cache initialized: {'MongoDB' if _cache_instance._use_mongo else 'In-memory'}")
    return _cache_instance

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
                cache = get_cache()  # Get cache once at the start
                
                if job_data.get('review_type') == 'general':
                    # --- GENERAL REVIEW ---
                    
                    # Get PR head commit SHA for file links (more reliable than branch name)
                    pr_data = fetcher.get_pr_threads(job_data['owner'], job_data['repo'], job_data['pr_number'])
                    head_commit_sha = pr_data.get("headRefOid", "") if pr_data else ""
                    head_ref = pr_data.get("headRefName", "main") if pr_data else "main"
                    
                    # Check cache first - CRITICAL for performance!
                    if head_commit_sha:
                        on_progress(f"🔍 Checking cache for commit {head_commit_sha[:8]}...")
                        logger.info(f"🔍 Checking cache for {job_data['owner']}/{job_data['repo']}#{job_data['pr_number']}@{head_commit_sha[:8]}")
                        cached_review = cache.get_general_review(
                            job_data['owner'], 
                            job_data['repo'], 
                            job_data['pr_number'], 
                            head_commit_sha
                        )
                        if cached_review:
                            on_progress(f"✅ Cache hit! Loading cached review...")
                            logger.info(f"✅ Cache hit for {job_data['owner']}/{job_data['repo']}#{job_data['pr_number']}@{head_commit_sha[:8]}")
                            review_id = f"{job_id}-general-review"
                            # Create a copy and add job-specific id
                            cached_review = cached_review.copy()
                            cached_review['id'] = review_id
                            
                            # Store in in-memory for follow-up questions
                            pr_files = fetcher.get_pr_files(job_data['owner'], job_data['repo'], job_data['pr_number'])
                            general_reviews[review_id] = {
                                "review_result": cached_review,
                                "owner": job_data['owner'],
                                "repo": job_data['repo'],
                                "pr_number": job_data['pr_number'],
                                "pr_files": pr_files,
                                "indexing_paths": job_data.get('indexing_paths')
                            }
                            
                            # Render HTML partial
                            html_content = templates.get_template("partials/general_result.html").render(
                                res=cached_review,
                                owner=job_data['owner'],
                                repo=job_data['repo'],
                                pr_number=job_data['pr_number'],
                                head_ref=head_ref,
                                head_commit_sha=head_commit_sha
                            )
                            payload = {
                                "id": review_id,
                                "file": cached_review.get('file', 'General Review'),
                                "status": cached_review.get('status', 'COMPLETED'),
                                "html": html_content
                            }
                            queue.put_nowait({"type": "result", "payload": payload})
                            on_progress("✅ General Review Complete (from cache)")
                            queue.put_nowait({"type": "done"})
                            return
                    
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
                    
                    on_progress(f"✓ Found {len(pr_files)} modified files")

                    async with git_sandbox(job_data['owner'], job_data['repo'], token, pr_number=job_data['pr_number'], on_progress=on_progress) as sandbox_dir:
                        # Setup Vector Store & Indexing for codebase context
                        vector_store = get_vector_store()
                        if vector_store:
                            # Pass cache to vector store for embedding caching
                            vector_store.set_cache(cache)
                            on_progress("📚 Step 2/4: Indexing codebase for context (this may take 30-60 seconds)...")
                            stats = await vector_store.index_repo(
                                sandbox_dir, 
                                include_paths=job_data.get('indexing_paths'), 
                                on_progress=on_progress,
                                owner=job_data['owner'],
                                repo=job_data['repo'],
                                pr_number=job_data['pr_number'],
                                commit_sha=head_commit_sha
                            )
                            queue.put_nowait({"type": "indexed", "stats": stats})
                            on_progress(f"✓ Indexed {stats.get('total_files', 0)} files")
                        
                        on_progress("🔬 Step 3/4: Launching parallel specialized analyses (5 AI agents)...")
                        
                        # General review doesn't use custom_prompt - it's a comprehensive review
                        result = await reviewer.general_pr_review(
                            pr_files, 
                            sandbox_dir, 
                            custom_prompt=None,  # General review is always comprehensive
                            retriever=vector_store,
                            indexing_paths=job_data.get('indexing_paths'),
                            on_progress=on_progress
                        )
                        
                        on_progress("📊 Step 4/4: Finalizing review report...")
                        
                        # Debug: Log result details
                        if result is not None:
                            logger.info(f"🔍 RESULT DEBUG: result type={type(result)}, is_dict={isinstance(result, dict)}, keys={list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                            print(f"🔍 RESULT DEBUG: result type={type(result)}, is_dict={isinstance(result, dict)}")
                            if isinstance(result, dict):
                                print(f"🔍 RESULT DEBUG: result keys={list(result.keys())}")
                                print(f"🔍 RESULT DEBUG: result['ai_analysis'] type={type(result.get('ai_analysis'))}")
                        else:
                            logger.warning(f"⚠️ RESULT DEBUG: result is None")
                            print(f"⚠️ RESULT DEBUG: result is None")
                        
                        if result and isinstance(result, dict) and result.get('ai_analysis'):
                            # IMPORTANT: Store in cache BEFORE modifying result
                            # Cache the original result structure with deep copy
                            logger.info(f"🔍 CACHE DEBUG: result is not None, head_commit_sha={head_commit_sha[:8] if head_commit_sha else 'EMPTY'}")
                            if head_commit_sha:
                                try:
                                    # Deep copy to avoid mutating nested structures
                                    cache_result = copy.deepcopy(result)
                                    logger.info(f"💾 CACHE: Preparing to cache: commit_sha={head_commit_sha[:8]}, result_keys={list(result.keys()) if isinstance(result, dict) else 'NOT_DICT'}")
                                    print(f"💾 CACHE: Attempting to cache review for {job_data['owner']}/{job_data['repo']}#{job_data['pr_number']}@{head_commit_sha[:8]}")
                                    
                                    cache_success = cache.store_general_review(
                                        job_data['owner'],
                                        job_data['repo'],
                                        job_data['pr_number'],
                                        head_commit_sha,
                                        cache_result,  # Store original result
                                        metadata={
                                            "indexing_paths": job_data.get('indexing_paths')
                                        }
                                    )
                                    
                                    print(f"💾 CACHE: store_general_review returned: {cache_success}")
                                    logger.info(f"💾 CACHE: store_general_review returned: {cache_success}")
                                    
                                    if cache_success:
                                        on_progress(f"💾 Cached review for commit {head_commit_sha[:8]}")
                                        logger.info(f"✅ Successfully cached general review for {job_data['owner']}/{job_data['repo']}#{job_data['pr_number']}@{head_commit_sha[:8]}")
                                        print(f"✅ CACHE SUCCESS: {job_data['owner']}/{job_data['repo']}#{job_data['pr_number']}@{head_commit_sha[:8]}")
                                    else:
                                        on_progress(f"⚠️ Warning: Failed to cache review (cache may be unavailable)")
                                        logger.error(f"❌ CACHE FAILED: Failed to cache review for {job_data['owner']}/{job_data['repo']}#{job_data['pr_number']}")
                                        print(f"❌ CACHE FAILED: {job_data['owner']}/{job_data['repo']}#{job_data['pr_number']}")
                                except Exception as e:
                                    import traceback
                                    logger.error(f"❌ CACHE EXCEPTION: {e}\n{traceback.format_exc()}")
                                    print(f"❌ CACHE EXCEPTION: {e}")
                                    on_progress(f"⚠️ Warning: Exception while caching review: {e}")
                            else:
                                on_progress(f"⚠️ Warning: No commit SHA available, review not cached")
                                logger.warning(f"⚠️ Cannot cache review: head_commit_sha is empty for {job_data['owner']}/{job_data['repo']}#{job_data['pr_number']}")
                                print(f"⚠️ CACHE SKIP: No commit SHA for {job_data['owner']}/{job_data['repo']}#{job_data['pr_number']}")
                        else:
                            logger.warning(f"⚠️ CACHE SKIP: result is None/empty for {job_data['owner']}/{job_data['repo']}#{job_data['pr_number']}")
                            print(f"⚠️ CACHE SKIP: result is None or empty")
                            if result is not None:
                                print(f"⚠️ CACHE SKIP DEBUG: result type={type(result)}, value={result}")
                        
                        # Store review data for follow-up questions (only if result exists)
                        if result and isinstance(result, dict) and result.get('ai_analysis'):
                            review_id = f"{job_id}-general-review"
                            
                            # Update result id to match review_id for template rendering (after caching)
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
                            # Try to get more details about why it failed
                            error_msg = "AI Review failed. Check server logs for details."
                            queue.put_nowait({"type": "error", "msg": error_msg})
                            on_progress("❌ Review returned no results. Check logs for error details.")

                else:
                    # --- COMMENT/THREAD ANALYSIS ---
                    cache = get_cache()
                    
                    # 1. Fetch All Comments
                    on_progress("🔍 Fetching all PR comments...")
                    # Update: get_pr_threads now returns a dict {items, headRefName, title, author}
                    pr_data = fetcher.get_pr_threads(job_data['owner'], job_data['repo'], job_data['pr_number'])
                    if not pr_data:
                        queue.put_nowait({"type": "error", "msg": "Could not fetch PR data"})
                        return
                        
                    all_items = pr_data.get("items", [])
                    head_ref = pr_data.get("headRefName", "main")
                    head_commit_sha = pr_data.get("headRefOid", "") if pr_data else ""
                    
                    # 2. Filter by selected IDs
                    selected_ids = set(job_data.get('selected_threads', []))
                    items_to_analyze = [item for item in all_items if item.get("id") in selected_ids]
                    
                    if not items_to_analyze:
                        queue.put_nowait({"type": "error", "msg": "No comments/threads selected"})
                        return

                    # Check cache for thread analyses
                    cached_results = []
                    uncached_items = []
                    if head_commit_sha:
                        for item in items_to_analyze:
                            thread_id = item.get("id")
                            cached = cache.get_thread_analysis(
                                job_data['owner'], 
                                job_data['repo'], 
                                job_data['pr_number'], 
                                head_commit_sha,
                                thread_id
                            )
                            if cached:
                                cached_results.append(cached)
                                on_progress(f"✅ Loaded cached analysis for {item.get('file_path', 'comment')}")
                            else:
                                uncached_items.append(item)
                    else:
                        uncached_items = items_to_analyze

                    # 3. Setup Sandbox & Indexing (only if we have uncached items)
                    vector_store = get_vector_store()
                    results_list = cached_results.copy()
                    
                    if uncached_items:
                        async with git_sandbox(job_data['owner'], job_data['repo'], token, pr_number=job_data['pr_number'], on_progress=on_progress) as sandbox_dir:
                            if vector_store:
                                # Pass cache to vector store for embedding caching
                                vector_store.set_cache(cache)
                                stats = await vector_store.index_repo(
                                    sandbox_dir, 
                                    include_paths=job_data['indexing_paths'], 
                                    on_progress=on_progress,
                                    owner=job_data['owner'],
                                    repo=job_data['repo'],
                                    pr_number=job_data['pr_number'],
                                    commit_sha=head_commit_sha
                                )
                                queue.put_nowait({"type": "indexed", "stats": stats})
                            
                            on_progress("🚀 Launching AI Agents...")
                            
                            # 4. Run Analysis in Parallel for uncached items
                            tasks = [
                                reviewer.analyze_thread(item, sandbox_dir, job_data.get('custom_prompt'), retriever=vector_store) 
                                for item in uncached_items
                            ]
                            
                            for future in asyncio.as_completed(tasks):
                                result = await future
                                if result:
                                    results_list.append(result)
                                    
                                    # Store in cache
                                    if head_commit_sha:
                                        thread_id = result.get('id', '')
                                        # Find the original item to get thread_id
                                        for item in uncached_items:
                                            if item.get('id') == thread_id or result.get('file') == item.get('file_path'):
                                                cache.store_thread_analysis(
                                                    job_data['owner'],
                                                    job_data['repo'],
                                                    job_data['pr_number'],
                                                    head_commit_sha,
                                                    item.get('id', thread_id),
                                                    result
                                                )
                                                break
                                    
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
                    else:
                        # All results were cached, just render them
                        for result in cached_results:
                            html_content = templates.get_template("partials/thread_result.html").render(res=result)
                            payload = {
                                "id": result['id'],
                                "file": result.get('file', 'General Comment'),
                                "status": result['status'],
                                "html": html_content
                            }
                            queue.put_nowait({"type": "result", "payload": payload})
                    
                    # 5. Generate Summary if we have results
                    if results_list:
                        # Check cache for summary
                        cached_summary = None
                        if head_commit_sha:
                            cached_summary = cache.get_analysis_summary(
                                job_data['owner'],
                                job_data['repo'],
                                job_data['pr_number'],
                                head_commit_sha
                            )
                        
                        if cached_summary:
                            on_progress("✅ Loaded cached summary")
                            summary = cached_summary
                        else:
                            on_progress("📊 Generating final resolution summary...")
                            summary = await reviewer.summarize_analyses(
                                results_list, 
                                job_data['owner'], 
                                job_data['repo'], 
                                job_data['pr_number'],
                                head_ref
                            )
                            
                            # Store summary in cache
                            if summary and head_commit_sha:
                                cache.store_analysis_summary(
                                    job_data['owner'],
                                    job_data['repo'],
                                    job_data['pr_number'],
                                    head_commit_sha,
                                    summary
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

@app.get("/analyze", response_class=HTMLResponse)
async def view_analysis_dashboard(
    request: Request,
    owner: str,
    repo: str,
    pr_number: int,
    review_type: str = "general"
):
    """GET endpoint to view analysis dashboard, auto-loads cached reviews if available"""
    import uuid
    job_id = str(uuid.uuid4()) if review_type == "general" else None
    
    # Render the dashboard shell - it will auto-load cached reviews
    return templates.TemplateResponse("analysis_dashboard.html", {
        "request": request,
        "job_id": job_id or "",
        "owner": owner,
        "repo": repo,
        "pr_number": pr_number
    })

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

@app.get("/api/prs/{owner}/{repo}/{pr_number}/cache-status")
async def get_cache_status(
    owner: str,
    repo: str,
    pr_number: int,
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    """Check if cached review is outdated by comparing commit SHAs"""
    try:
        cache = get_cache()
        
        # Get current PR commit SHA
        pr_data = fetcher.get_pr_threads(owner, repo, pr_number)
        if not pr_data:
            return {"error": "Could not fetch PR data"}
        
        current_commit_sha = pr_data.get("headRefOid", "")
        if not current_commit_sha:
            return {"error": "Could not get current commit SHA"}
        
        logger.debug(f"🔍 get_cache_status: Checking cache for {owner}/{repo}#{pr_number}@{current_commit_sha[:8]}")
        
        # Check cache for current commit first
        cached_review = cache.get_general_review(owner, repo, pr_number, current_commit_sha)
        has_cache_for_current = cached_review is not None
        
        logger.debug(f"🔍 get_cache_status: Cache check for current commit: {has_cache_for_current}")
        
        # Get latest cached commit
        latest_cached = cache.get_latest_cached_commit(owner, repo, pr_number)
        
        logger.debug(f"🔍 get_cache_status: Latest cached commit: {latest_cached.get('commit_sha', 'None')[:8] if latest_cached else 'None'}")
        
        if not latest_cached and not has_cache_for_current:
            logger.debug(f"🔍 get_cache_status: No cache found for {owner}/{repo}#{pr_number}")
            return {
                "has_cache": False,
                "is_outdated": False,
                "current_commit_sha": current_commit_sha,
                "cached_commit_sha": None,
                "cache_type": "MongoDB" if cache._use_mongo else "In-memory"
            }
        
        cached_commit_sha = latest_cached.get("commit_sha", "") if latest_cached else current_commit_sha
        is_outdated = cached_commit_sha != current_commit_sha if latest_cached else False
        
        logger.info(f"✅ get_cache_status: {owner}/{repo}#{pr_number} - has_cache={True}, is_outdated={is_outdated}, current={current_commit_sha[:8]}, cached={cached_commit_sha[:8] if cached_commit_sha else 'None'}")
        
        return {
            "has_cache": True,
            "is_outdated": is_outdated,
            "current_commit_sha": current_commit_sha,
            "cached_commit_sha": cached_commit_sha,
            "cached_at": latest_cached.get("created_at").isoformat() if latest_cached and latest_cached.get("created_at") else None,
            "cache_type": "MongoDB" if cache._use_mongo else "In-memory"
        }
    except Exception as e:
        import traceback
        logger.error(f"Error checking cache status: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}

@app.get("/api/prs/{owner}/{repo}/{pr_number}/cached-review")
async def get_cached_review(
    owner: str,
    repo: str,
    pr_number: int,
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    """Get cached general review if available"""
    try:
        cache = get_cache()
        
        # Get current PR commit SHA
        pr_data = fetcher.get_pr_threads(owner, repo, pr_number)
        if not pr_data:
            raise HTTPException(status_code=404, detail="Could not fetch PR data")
        
        current_commit_sha = pr_data.get("headRefOid", "")
        if not current_commit_sha:
            raise HTTPException(status_code=400, detail="Could not get current commit SHA")
        
        # Get cached review
        cached_review = cache.get_general_review(owner, repo, pr_number, current_commit_sha)
        
        if not cached_review:
            # Try to get latest cached commit to see what we have
            latest_cached = cache.get_latest_cached_commit(owner, repo, pr_number)
            if latest_cached:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No cached review found for commit {current_commit_sha[:8]}. Latest cached commit: {latest_cached.get('commit_sha', 'unknown')[:8]}"
                )
            raise HTTPException(status_code=404, detail="No cached review found")
        
        head_ref = pr_data.get("headRefName", "main")
        
        # Render HTML
        html_content = templates.get_template("partials/general_result.html").render(
            res=cached_review,
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            head_ref=head_ref,
            head_commit_sha=current_commit_sha
        )
        
        return {
            "review": cached_review,
            "html": html_content
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Error getting cached review: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prs/{owner}/{repo}/{pr_number}/regenerate")
async def force_regenerate(
    owner: str,
    repo: str,
    pr_number: int,
    request: Request,
    fetcher: GitHubFetcher = Depends(get_github_fetcher)
):
    """Force re-generation by clearing cache for current commit"""
    try:
        cache = get_cache()
        
        # Get current PR commit SHA
        pr_data = fetcher.get_pr_threads(owner, repo, pr_number)
        if not pr_data:
            raise HTTPException(status_code=404, detail="Could not fetch PR data")
        
        current_commit_sha = pr_data.get("headRefOid", "")
        if not current_commit_sha:
            raise HTTPException(status_code=400, detail="Could not get current commit SHA")
        
        # Clear cache for current commit
        cache.clear_commit_cache(owner, repo, pr_number, current_commit_sha)
        
        return {
            "success": True,
            "message": "Cache cleared. Please re-run the analysis.",
            "commit_sha": current_commit_sha
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
