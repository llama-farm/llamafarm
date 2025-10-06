#!/usr/bin/env python3
"""
Test script for the ProjectsTool
"""

import sys
import os
from pathlib import Path

# Add the current directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from tool.projects_tool import ProjectsTool, ProjectsToolInput

def test_projects_tool():
    """Test the ProjectsTool functionality."""
    print("Testing ProjectsTool...")
    
    # Create tool instance
    tool = ProjectsTool()
    
    # Test listing projects
    print("\n1. Testing list projects:")
    result = tool.run(ProjectsToolInput(action="list", namespace="test"))
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")
    print(f"   Total projects: {result.total}")
    if result.projects:
        for project in result.projects:
            print(f"   - {project['namespace']}/{project['project_id']}")
    
    # Test creating a project
    print("\n2. Testing create project:")
    result = tool.run(
        ProjectsToolInput(action="create", namespace="test", project_id="test_project")
    )
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")
    if result.projects:
        for project in result.projects:
            print(f"   - Created: {project['namespace']}/{project['project_id']}")
    
    # Test creating the same project again (should fail)
    print("\n3. Testing create duplicate project:")
    result = tool.run(
        ProjectsToolInput(action="create", namespace="test", project_id="test_project")
    )
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")
    
    # Test listing projects again (should now have one project)
    print("\n4. Testing list projects after creation:")
    result = tool.run(ProjectsToolInput(action="list", namespace="test"))
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")
    print(f"   Total projects: {result.total}")
    if result.projects:
        for project in result.projects:
            print(f"   - {project['namespace']}/{project['project_id']}")
    
    # Test invalid action (this will raise a validation error)
    print("\n5. Testing invalid action (validation error expected):")
    try:
        result = tool.run(ProjectsToolInput(action="invalid", namespace="test"))
        print(f"   Success: {result.success}")
        print(f"   Message: {result.message}")
    except Exception as e:
        print(f"   Expected validation error: {type(e).__name__}: {str(e)}")
    
    # Test create without project_id
    print("\n6. Testing create without project_id:")
    result = tool.run(ProjectsToolInput(action="create", namespace="test"))
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_projects_tool() 