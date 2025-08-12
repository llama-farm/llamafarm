#!/usr/bin/env python3
"""
Automatically update contributors section in README.md from GitHub API.

This script fetches contributor information from the GitHub API and updates
the contributors section in the README.md file using the all-contributors format.

Usage:
    python scripts/update_contributors.py [--token YOUR_GITHUB_TOKEN]
    
Environment Variables:
    GITHUB_TOKEN: GitHub personal access token for API rate limits
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError

def fetch_contributors(
    owner: str = "llama-farm",
    repo: str = "llamafarm",
    token: Optional[str] = None
) -> List[Dict]:
    """
    Fetch contributors from GitHub API.
    
    Args:
        owner: GitHub organization or user
        repo: Repository name
        token: Optional GitHub token for increased rate limits
        
    Returns:
        List of contributor dictionaries
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "LlamaFarm-Contributor-Bot"
    }
    
    if token:
        headers["Authorization"] = f"token {token}"
    
    try:
        req = Request(url, headers=headers)
        with urlopen(req) as response:
            return json.loads(response.read().decode())
    except HTTPError as e:
        if e.code == 404:
            print(f"Repository {owner}/{repo} not found")
        elif e.code == 403:
            print("Rate limit exceeded. Please provide a GitHub token.")
        else:
            print(f"Error fetching contributors: {e}")
        return []

def fetch_user_details(username: str, token: Optional[str] = None) -> Dict:
    """
    Fetch detailed user information from GitHub API.
    
    Args:
        username: GitHub username
        token: Optional GitHub token
        
    Returns:
        User details dictionary
    """
    url = f"https://api.github.com/users/{username}"
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "LlamaFarm-Contributor-Bot"
    }
    
    if token:
        headers["Authorization"] = f"token {token}"
    
    try:
        req = Request(url, headers=headers)
        with urlopen(req) as response:
            return json.loads(response.read().decode())
    except HTTPError:
        return {}

def generate_contributor_html(contributors: List[Dict], token: Optional[str] = None) -> str:
    """
    Generate HTML table for contributors in all-contributors format.
    
    Args:
        contributors: List of contributor data
        token: Optional GitHub token
        
    Returns:
        HTML string for contributor table
    """
    if not contributors:
        return ""
    
    html_parts = [
        "<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->",
        "<!-- prettier-ignore-start -->",
        "<!-- markdownlint-disable -->",
        "<table>",
        "  <tbody>"
    ]
    
    # Process contributors in rows of 7
    row_size = 7
    for i in range(0, len(contributors), row_size):
        html_parts.append("    <tr>")
        
        for contributor in contributors[i:i+row_size]:
            username = contributor.get("login", "")
            avatar = contributor.get("avatar_url", "")
            profile = contributor.get("html_url", "")
            contributions = contributor.get("contributions", 0)
            
            # Fetch additional user details
            user_details = fetch_user_details(username, token)
            name = user_details.get("name") or username
            
            # Generate contribution badges
            badges = []
            if contributions >= 10:
                badges.append(f'<a href="https://github.com/llama-farm/llamafarm/commits?author={username}" title="Code">💻</a>')
            if contributions >= 50:
                badges.append('<a href="#maintenance" title="Maintenance">🚧</a>')
            if contributions >= 100:
                badges.append('<a href="#projectManagement" title="Project Management">📆</a>')
            
            # Always include at least the code contribution badge
            if not badges:
                badges.append(f'<a href="https://github.com/llama-farm/llamafarm/commits?author={username}" title="Code">💻</a>')
            
            cell_html = f'''      <td align="center" valign="top" width="14.28%">
        <a href="{profile}">
          <img src="{avatar}?v=4&s=100" width="100px;" alt="{name}"/>
          <br />
          <sub><b>{name}</b></sub>
        </a>
        <br />
        {" ".join(badges)}
      </td>'''
            
            html_parts.append(cell_html)
        
        html_parts.append("    </tr>")
    
    html_parts.extend([
        "  </tbody>",
        "</table>",
        "<!-- markdownlint-restore -->",
        "<!-- prettier-ignore-end -->",
        "<!-- ALL-CONTRIBUTORS-LIST:END -->"
    ])
    
    return "\n".join(html_parts)

def update_readme(contributor_html: str, readme_path: Path = Path("README.md")) -> bool:
    """
    Update the README.md file with new contributor HTML.
    
    Args:
        contributor_html: HTML string for contributors
        readme_path: Path to README.md file
        
    Returns:
        True if successful, False otherwise
    """
    if not readme_path.exists():
        print(f"README.md not found at {readme_path}")
        return False
    
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find and replace the contributor section
    pattern = r"<!-- ALL-CONTRIBUTORS-LIST:START.*?<!-- ALL-CONTRIBUTORS-LIST:END -->"
    
    if re.search(pattern, content, re.DOTALL):
        # Replace existing section
        new_content = re.sub(pattern, contributor_html, content, flags=re.DOTALL)
    else:
        # Find the Contributors heading and add after it
        contrib_pattern = r"(### 🏆 Contributors\n+)"
        if re.search(contrib_pattern, content):
            new_content = re.sub(
                contrib_pattern,
                f"\\1{contributor_html}\n\n",
                content
            )
        else:
            print("Could not find Contributors section in README")
            return False
    
    # Write updated content
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    return True

def main():
    """Main function to update contributors."""
    # Get GitHub token from environment or command line
    token = os.environ.get("GITHUB_TOKEN")
    
    if len(sys.argv) > 1 and sys.argv[1].startswith("--token"):
        if "=" in sys.argv[1]:
            token = sys.argv[1].split("=")[1]
        elif len(sys.argv) > 2:
            token = sys.argv[2]
    
    # Find project root (where README.md is)
    current_dir = Path.cwd()
    readme_path = current_dir / "README.md"
    
    # If not in root, try parent directory
    if not readme_path.exists():
        readme_path = current_dir.parent / "README.md"
    
    if not readme_path.exists():
        print("Could not find README.md. Please run from project root.")
        sys.exit(1)
    
    print("🦙 LlamaFarm Contributor Updater")
    print("-" * 40)
    
    # Fetch contributors
    print("Fetching contributors from GitHub...")
    contributors = fetch_contributors(token=token)
    
    if not contributors:
        print("No contributors found or error fetching data.")
        print("Note: The repository might not exist yet or be private.")
        print("\nGenerating placeholder for future contributors...")
        
        # Generate placeholder
        contributor_html = """<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/llama-farm">
          <img src="https://avatars.githubusercontent.com/u/123456789?v=4&s=100" width="100px;" alt="LlamaFarm Team"/>
          <br />
          <sub><b>LlamaFarm Team</b></sub>
        </a>
        <br />
        <a href="https://github.com/llama-farm/llamafarm" title="Code">💻</a>
        <a href="#maintenance" title="Maintenance">🚧</a>
        <a href="#projectManagement" title="Project Management">📆</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <i>Your contribution here!</i>
        <br />
        <sub><b>Join us!</b></sub>
        <br />
        <a href="https://github.com/llama-farm/llamafarm/blob/main/CONTRIBUTING.md" title="Contributing">🤝</a>
      </td>
    </tr>
  </tbody>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->"""
    else:
        print(f"Found {len(contributors)} contributors")
        
        # Sort by contributions
        contributors.sort(key=lambda x: x.get("contributions", 0), reverse=True)
        
        # Limit to top 50 contributors for README
        if len(contributors) > 50:
            print(f"Showing top 50 contributors (total: {len(contributors)})")
            contributors = contributors[:50]
        
        # Generate HTML
        print("Generating contributor HTML...")
        contributor_html = generate_contributor_html(contributors, token)
    
    # Update README
    print("Updating README.md...")
    if update_readme(contributor_html, readme_path):
        print("✅ Successfully updated contributors in README.md")
        
        # Show statistics
        if contributors:
            total_contributions = sum(c.get("contributions", 0) for c in contributors)
            print(f"\nStatistics:")
            print(f"  Total contributors shown: {len(contributors)}")
            print(f"  Total contributions: {total_contributions}")
            print(f"  Top contributor: {contributors[0].get('login')} ({contributors[0].get('contributions')} contributions)")
    else:
        print("❌ Failed to update README.md")
        sys.exit(1)

if __name__ == "__main__":
    main()