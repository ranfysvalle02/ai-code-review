# AI Open-Sourcerer

## Code is Cheap. Context is Expensive.

![](wrking-results-go.png)

**AI Open-Sourcerer** is a local "Context Engine" designed to help developers review and resolve complex GitHub Pull Request threads. It acts as a "Glass Box" tool that clones the repository, indexes the code semantically using Voyage AI, and uses OpenAI's GPT-4o to analyze discussions, validate code against tests, and propose specific fixes.

We have entered a rapid-cycle era of engineering. With modern AI tooling, the time it takes to validate an idea or prototype a feature has collapsed. Syntactically, code has never been cheaper to write.

But while we can generate logic instantly, **context** remains stubbornly expensive.

My resolution for 2026 is to contribute significantly more to open source. But I’ve found that the bottleneck isn't writing the code—it's navigating the review.

We’ve all been there. You find a high-impact repository. You spot a Pull Request that is 90% there but stuck in limbo. It’s not broken, but it’s blocked by five different conversation threads—debates about naming, edge cases, or test expectations. To help move it forward, you have to spend hours acting as a digital archaeologist, context-switching between threads just to offer a valid opinion on one of them.

The constraint isn't keystrokes; it's clarity.

**AI Open-Sourcerer** is a prototype designed to solve this specific friction. It is a local "Context Engine" that allows me to drop into a specific thread of a massive PR and resolve it immediately, without the ramp-up tax.

## Features

- **"Glass Box" Analysis**: Clones the repository into an ephemeral sandbox to analyze the actual code state, not just the diff.
- **Semantic Context (RAG)**: Uses [Voyage AI](https://voyageai.com/) (`voyage-code-2`) to index the codebase and retrieve relevant dependencies for a given discussion thread.
- **Structured Reasoning**: Uses OpenAI Structured Outputs (Pydantic) to force the AI to "think" step-by-step, analyze the debate, and check for race conditions or architectural mismatches.
- **Interactive Dashboard**: A FastAPI + HTMX web interface to browse PRs, select threads, and stream analysis results in real-time.
- **Smart Context**: Automatically suggests which files and documentation to index based on the PR's modified files.

## Prerequisites

- Python 3.8+
- [GitHub Personal Access Token](https://github.com/settings/tokens) (with `repo` scope)
- [OpenAI API Key](https://platform.openai.com/)
- [Voyage AI API Key](https://voyageai.com/) (for semantic search)

## Usage

1. **Start the Application:**
   ```bash
   uvicorn main:app --reload
   ```

2. **Access the Web Interface:**
   Open your browser and navigate to `http://localhost:8000`.

3. **Workflow:**
   - **Search:** Enter a GitHub repository owner and name (e.g., `fastapi/fastapi`).
   - **Select PR:** Choose a Pull Request from the list.
   - **Context:** The tool will suggest relevant files to index. You can adjust this selection.
   - **Analyze:** Select specific discussion threads you want to resolve and click "Start Analysis".
   - **Review:** Watch as the AI agents analyze each thread in parallel and generate a detailed report with proposed fixes.


-----

## The Parable of the Stale PR

![](working-1.png)

PRs rarely get stuck "as a whole." They get stuck in the details—a specific disagreement on line 402 that stalls the entire merge for weeks.

To test this, I pointed the tool at a stalled PR in a widely used **Database Exporter** project. The code had been sitting untouched for months.

The discussion was stuck on a subtle mismatch between the implementation (filtering database views) and the unit test expectations. The PR had stalled because the maintainer and the contributor were talking past each other: the contributor thought the test should expect an error, while the maintainer implied it should just fail silently.

The AI didn't just summarize the chat; it audited the test file against the source code, realized the test was asserting the wrong behavior, and provided the specific code to resolve **that specific thread**.

**Here is the actual output from the dashboard:**

> **FILE:** `pkg/exporter/common_test.go`
> **STATUS:** `UNRESOLVED` | **SEVERITY:** `MEDIUM`
>
> **Summary:** Test names and expected results in error handling of view filtering need updating to match recent code changes.
>
> **Technical Deep Dive:**
>
> 1.  **Code Analysis:** The test `TestCheckNamespaces` incorrectly expects an error when a "View" is encountered. The actual function was updated in this PR to silently exclude views and log a warning instead.
> 2.  **Debate Evaluation:** The maintainer correctly pointed out that only collections should be returned. The test is asserting an outdated contract.
> 3.  **Resolution Critique:** The code is correct, but the test suite is failing because it wasn't updated to reflect the new "silent exclusion" strategy.
>
> **Proposed Fix:**
> *"Update test naming and adjust assertions to reflect logic that views are simply not returned."*
>
> ```go
> t.Run("Views or non-exist namespace in provided collection list (should fail)", func(t *testing.T) {
>     // We expect NO error, but the 'view01' should be missing from the result
>     filtered, err := checkNamespacesForViewsOrNonExist(ctx, client, []string{"testdb01.col01", "testdb01.system.views", "testdb01.view01"}, nil)
>     assert.NoError(t, err)
>     assert.Equal(t, []string{"testdb01.col01"}, filtered)
> })
> ```

This is the power of the tool. It doesn't magically "fix the PR" in one shot. It acts as a surgical instrument, allowing me to step into a complex debate, provide the exact code needed to resolve that thread, and unblock the maintainers to focus on the next one.

-----

## The Architecture: "Glass Box" Design

Most current AI tools act as "Black Boxes"—they want to take the wheel and drive. That works for greenfield scripts, but it fails for complex, existing systems where architectural nuances matter.

I needed a researcher, not a replacement. I built this using three specific strategies to ensure transparency and speed.

### 1\. Ephemeral Sandboxing (`asyncio` + `tempfile`)

Security and cleanliness are paramount when auditing third-party code. I didn't want a tool that messes with my local environment.

I implemented an **Async Context Manager** that treats every analysis as an ephemeral event. It clones the repo into a temporary directory, performs the heavy lifting, and destroys the directory the moment the context exits.

```python
@asynccontextmanager
async def git_sandbox(owner: str, repo: str, token: str, pr_number: int = None, on_progress: Callable = None):
    # Create a unique, temporary directory
    sandbox_dir = tempfile.mkdtemp(prefix=f"sandbox_{owner}_{repo}_")
    try:
        # Clone heavily optimized (depth 1) for speed
        clone_url = f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"
        
        # Async subprocess execution to keep the UI responsive
        await asyncio.to_thread(run_git, ["git", "clone", "--depth", "1", clone_url, sandbox_dir])
        
        # ... fetch PR logic ...
        yield sandbox_dir
    finally:
        # Automatic cleanup guarantees no disk clutter
        if os.path.exists(sandbox_dir):
            await asyncio.to_thread(shutil.rmtree, sandbox_dir)
```

### 2\. Semantic Indexing (Voyage AI)

Standard RAG (Retrieval Augmented Generation) often fails on code because it treats it like English prose. It matches keywords but misses the *structure*.

I integrated **Voyage AI** (specifically `voyage-code-2`) to build a semantic index. This model understands programming concepts—imports, class hierarchies, and dependency injections. When the AI analyzes a thread, it retrieves not just the file that changed, but the *dependencies* of that file, providing a complete architectural picture.

### 3\. Structured Reasoning (Pydantic)

LLMs are stochastic; engineering requires precision. I didn't want the AI to "chat" with me. I wanted it to think.

I used **Pydantic** to enforce a strict output schema. The AI cannot just return text; it must fill out a `CodeSuggestion` object containing specific fields for `debate_analysis`, `race_condition_checks`, and a step-by-step `thinking_process`.

```python
class CodeSuggestion(BaseModel):
    summary: str = Field(..., description="A 1-sentence summary of the disagreement.")
    severity: str = Field(..., description="Criticality: Low, Medium, High")
    
    # We force the model to 'think' step-by-step before concluding
    thinking_process: List[str] = Field(..., description="Step-by-step reasoning.")
    
    debate_analysis: str = Field(..., description="Analysis of arguments using first principles.")
    proposed_fix: Optional[str] = Field(None, description="The exact code fix, if needed.")

# ... inside the analysis loop ...
completion = await client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format=CodeSuggestion, # <--- The magic happens here
)
```

-----

## The Future: Fundamentals + Guardrails

This project is currently a "Read-Only" tool—it helps me understand the code so I can help fix it. But it points toward how our industry is evolving.

We are entering the era of **"Vibe Coding"**—where developers focus on flow and intent, letting the AI handle implementation details. Right now, this workflow is in the worst state it's ever going to be. It feels messy. We generate code faster than ever, but we often drown in "almost-right" architecture that passes unit tests but breaks design patterns.

But the industry is adapting. We are starting to see the emergence of **codified context**.

Tools like **Cursor** have recently introduced **Project Rules** (via `.cursor/rules` or `AGENTS.md`), which allow maintainers to bundle system-level instructions, prompts, and architecture decisions directly into the repository. This is a massive shift: it means "context" is becoming a checked-in artifact, just like your code.

**AI Open-Sourcerer** builds on this philosophy. It’s not just about generating code; it's about retrieving that specific, codified context and using it as a guardrail.

Imagine a workflow where you prototype a feature in an afternoon. Instead of spending three days fighting with a linter or arguing about style, you hand it off to an agent that:

1.  **Retrieves** the project's specific `CONTRIBUTING.md`, style guides, and `.cursor/rules`.
2.  **Refactors** your "vibe code" to match the strict patterns of the repository.
3.  **Generates** the documentation the maintainers actually want.

It’s just about using AI to clean up the mess that AI creates.

**AI Open-Sourcerer** is my attempt to automate that cleanup. If you want to join me in making 2026 a year of high-impact contribution, the code is open.

-----

### Appendix: The Stack

For those interested in the implementation details, here is the tech stack that powers the engine:

  * **Backend:** FastAPI (Python)
  * **Concurrency:** `asyncio` for parallel agent execution.
  * **Embeddings:** Voyage AI (`voyage-code-2`) for semantic code search.
  * **Reasoning:** OpenAI (`gpt-4o`) with Structured Outputs (Pydantic).
  * **Frontend:** HTMX + Jinja2 (No React/Vue overhead, just pure HTML over the wire).
  * **Environment:** Ephemeral `tempfile` directories for security.

---

## Project Structure

- `main.py`: FastAPI application entry point and route definitions.
- `services.py`: Core business logic including:
  - `GitHubFetcher`: GraphQL client for GitHub API.
  - `git_sandbox`: Async context manager for temporary repo cloning.
  - `VectorStore`: Voyage AI integration for semantic indexing.
  - `AIReviewer`: GPT-4o integration with structured outputs.
- `templates/`: HTML templates for the UI.

## Legacy CLI

A legacy CLI version is available in `cli_legacy.py`, though the web interface is the recommended way to use the tool.
