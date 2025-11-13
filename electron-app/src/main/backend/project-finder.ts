/**
 * Project Finder - Discovers or initializes LlamaFarm projects
 */

import { promises as fsPromises } from 'fs'
import * as path from 'path'
import * as fs from 'fs'
import { app } from 'electron'
import { exec } from 'child_process'
import { promisify } from 'util'

const execAsync = promisify(exec)

export interface Project {
  path: string
  name: string
  namespace: string
}

export class ProjectFinder {
  private readonly defaultProjectsDir: string

  constructor() {
    // Default LlamaFarm projects location
    this.defaultProjectsDir = path.join(app.getPath('home'), '.llamafarm', 'projects')
  }

  /**
   * Find all available projects
   */
  async findProjects(): Promise<Project[]> {
    const projects: Project[] = []

    try {
      // Check if projects directory exists
      if (!fs.existsSync(this.defaultProjectsDir)) {
        return []
      }

      // Scan for projects (namespace/project structure)
      const namespaces = await fsPromises.readdir(this.defaultProjectsDir)

      for (const namespace of namespaces) {
        const namespacePath = path.join(this.defaultProjectsDir, namespace)
        const stat = await fsPromises.stat(namespacePath)

        if (!stat.isDirectory()) continue

        // Check each project in the namespace
        const projectNames = await fsPromises.readdir(namespacePath)

        for (const projectName of projectNames) {
          const projectPath = path.join(namespacePath, projectName)
          const projectStat = await fsPromises.stat(projectPath)

          if (!projectStat.isDirectory()) continue

          // Check if it has a llamafarm.yaml file
          const configPath = path.join(projectPath, 'llamafarm.yaml')
          if (fs.existsSync(configPath)) {
            projects.push({
              path: projectPath,
              name: projectName,
              namespace
            })
          }
        }
      }
    } catch (error) {
      console.error('Error finding projects:', error)
    }

    return projects
  }

  /**
   * Get or create default project
   */
  async getOrCreateDefaultProject(cliPath: string): Promise<string> {
    // First, try to find existing projects
    const projects = await this.findProjects()

    if (projects.length > 0) {
      // Return the first project found
      console.log(`Found existing project: ${projects[0].namespace}/${projects[0].name}`)
      return projects[0].path
    }

    // No projects found, create a default one
    console.log('No projects found, initializing default project...')
    return await this.createDefaultProject(cliPath)
  }

  /**
   * Create a default project
   */
  private async createDefaultProject(cliPath: string): Promise<string> {
    const projectName = 'desktop-project'
    const namespace = 'default'
    const projectPath = path.join(this.defaultProjectsDir, namespace, projectName)

    try {
      // Ensure parent directory exists
      await fsPromises.mkdir(path.join(this.defaultProjectsDir, namespace), { recursive: true })

      // Run lf init
      console.log(`Creating project at: ${projectPath}`)
      await execAsync(`"${cliPath}" init ${projectName} --namespace ${namespace}`, {
        cwd: path.join(this.defaultProjectsDir, namespace)
      })

      console.log(`Project created successfully at: ${projectPath}`)
      return projectPath
    } catch (error) {
      console.error('Failed to create default project:', error)
      throw new Error(`Failed to create default project: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  /**
   * Check if a directory has a LlamaFarm project
   */
  async isProjectDirectory(dir: string): Promise<boolean> {
    try {
      const configPath = path.join(dir, 'llamafarm.yaml')
      return fs.existsSync(configPath)
    } catch {
      return false
    }
  }
}
