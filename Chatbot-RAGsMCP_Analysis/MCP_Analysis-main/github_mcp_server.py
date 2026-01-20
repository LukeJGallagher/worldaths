import os
import argparse
import json
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from github import Github
from dotenv import load_dotenv
from openrouter_utils import github_analyze

# Parse command-line arguments
parser = argparse.ArgumentParser(description="GitHub API Server")
parser.add_argument("--port", type=int, default=8002, help="Port to run the server on")
args = parser.parse_args()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app - simple definition to avoid issues
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GitHub configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if GITHUB_TOKEN:
    logger.info("GitHub token found")
else:
    logger.warning("No GitHub token found in environment variables")
g = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None

# Root endpoint
@app.get("/")
def root():
    logger.info("Root endpoint called")
    return {
        "status": "running",
        "message": "GitHub Explorer API is running"
    }

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "github_client": "connected" if g else "disconnected"
    }

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests to the server"""
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

# List repositories endpoint
@app.get("/repos")
async def list_repositories():
    """List all accessible repositories"""
    if not g:
        return {"error": "GitHub token not configured"}
    
    try:
        repos = []
        for repo in g.get_user().get_repos():
            repos.append({
                "name": repo.name,
                "description": repo.description,
                "url": repo.html_url,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count
            })
        return repos
    except Exception as e:
        return {"error": str(e)}

# Get repository details endpoint
@app.get("/repo/{repo_name}")
async def get_repository(repo_name: str):
    """Get details about a specific repository"""
    if not g:
        return {"error": "GitHub token not configured"}
    
    try:
        repo = g.get_user().get_repo(repo_name)
        return {
            "name": repo.name,
            "description": repo.description,
            "url": repo.html_url,
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "language": repo.language,
            "topics": repo.get_topics(),
            "last_updated": repo.updated_at.isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

# List repository files endpoint
@app.get("/repo/{repo_name}/files")
async def list_repo_files(repo_name: str):
    """List files in a repository"""
    if not g:
        return {"error": "GitHub token not configured"}
    
    try:
        repo = g.get_user().get_repo(repo_name)
        contents = repo.get_contents("")
        files = []
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(repo.get_contents(file_content.path))
            else:
                files.append({
                    "path": file_content.path,
                    "type": file_content.type,
                    "size": file_content.size,
                    "url": file_content.html_url
                })
        return files
    except Exception as e:
        return {"error": str(e)}

# Search repositories endpoint
@app.get("/search")
async def search_repositories(query: str):
    """Search repositories"""
    if not g:
        return {"error": "GitHub token not configured"}
    
    try:
        repos = g.search_repositories(query)
        results = []
        for repo in repos[:10]:  # Limit to 10 results
            results.append({
                "name": repo.name,
                "description": repo.description,
                "url": repo.html_url,
                "stars": repo.stargazers_count
            })
        return results
    except Exception as e:
        return {"error": str(e)}

# Get file content endpoint
@app.get("/file/{repo_name}/{file_path}")
async def get_file_content(repo_name: str, file_path: str):
    """Get content of a specific file"""
    if not g:
        return {"error": "GitHub token not configured"}
    
    try:
        repo = g.get_user().get_repo(repo_name)
        content = repo.get_contents(file_path)
        return content.decoded_content.decode('utf-8')
    except Exception as e:
        return {"error": str(e)}

# Analyze repository endpoint
@app.get("/analyze/{repo_name}")
async def analyze_repository(repo_name: str):
    """Analyze a repository using OpenRouter"""
    if not g:
        return {"error": "GitHub token not configured"}
    
    try:
        # First get repository details to provide context
        repo = g.get_user().get_repo(repo_name)
        repo_info = {
            "name": repo.name,
            "description": repo.description,
            "language": repo.language,
            "topics": repo.get_topics(),
            "stars": repo.stargazers_count,
            "forks": repo.forks_count
        }
        
        # Get a list of important files
        contents = repo.get_contents("")
        files = []
        while contents and len(files) < 100:  # Limit to avoid processing too many files
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(repo.get_contents(file_content.path))
            else:
                files.append(file_content.path)
        
        # Create a task description for the LLM
        task = f"""
        Analyze the GitHub repository with the following details:
        
        Repository: {json.dumps(repo_info, indent=2)}
        
        Key files: {json.dumps(files[:20], indent=2)}  # Limited to 20 for brevity
        
        Tasks:
        1. Summarize the purpose of this repository
        2. Identify the main programming language and technologies used
        3. Note any documentation or README content
        4. Suggest improvements if applicable
        5. Rate the project on documentation, structure, and maintainability
        """
        
        # Use OpenRouter to analyze
        analysis = github_analyze(repo_name, task)
        
        # Return both the raw data and the analysis
        return {
            "repository_info": repo_info,
            "key_files": files[:20],
            "analysis": analysis.get("analysis", "Analysis failed"),
            "error": analysis.get("error", None)
        }
    except Exception as e:
        return {"error": str(e)}

# Find similar repositories endpoint
@app.get("/find-similar/{repo_name}")
async def find_similar_repositories(repo_name: str):
    """Find similar repositories using OpenRouter"""
    if not g:
        return {"error": "GitHub token not configured"}
    
    try:
        # Get repository details to provide context
        repo = g.get_user().get_repo(repo_name)
        
        # Create a task description for the LLM
        task = f"""
        Find repositories similar to this one:
        
        Repository: {repo.name}
        Description: {repo.description}
        Language: {repo.language}
        Topics: {repo.get_topics()}
        
        Tasks:
        1. Identify the key characteristics of this repository
        2. Suggest search terms for finding similar repositories
        3. Recommend 5-10 similar repositories based on the description and topics
        4. Explain why each recommendation might be useful to someone interested in {repo.name}
        """
        
        # Use OpenRouter to analyze
        analysis = github_analyze(repo_name, task)
        
        # Return both the raw data and the analysis
        return {
            "repository": repo.name,
            "language": repo.language,
            "topics": repo.get_topics(),
            "analysis": analysis.get("analysis", "Analysis failed"),
            "error": analysis.get("error", None)
        }
    except Exception as e:
        return {"error": str(e)}

# Main entry point
if __name__ == "__main__":
    PORT = int(os.getenv("GITHUB_SERVER_PORT", args.port))
    logger.info(f"Starting GitHub server on http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT) 