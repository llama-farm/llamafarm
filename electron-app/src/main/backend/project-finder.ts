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
  private readonly lastProjectFile: string

  constructor() {
    // Default LlamaFarm projects location
    this.defaultProjectsDir = path.join(app.getPath('home'), '.llamafarm', 'projects')

    // Store last used project in app's userData
    this.lastProjectFile = path.join(app.getPath('userData'), 'last-project.txt')
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
   * On first launch: picks a random valid project
   * On subsequent launches: uses the last project (Designer can change this)
   */
  async getOrCreateDefaultProject(cliPath: string): Promise<string> {
    // Check if we have a last used project
    const lastProject = await this.getLastUsedProject()
    if (lastProject && fs.existsSync(lastProject)) {
      console.log(`Using last project: ${lastProject}`)
      return lastProject
    }

    // First launch - find all valid projects
    console.log('First launch - scanning for valid projects...')
    const projects = await this.findProjects()

    // Filter to valid projects only
    const validProjects: Project[] = []
    for (const project of projects) {
      if (await this.isValidProject(project.path)) {
        validProjects.push(project)
      }
    }

    if (validProjects.length > 0) {
      // Pick a random valid project
      const randomIndex = Math.floor(Math.random() * validProjects.length)
      const chosenProject = validProjects[randomIndex]

      console.log(`✓ Selected random project: ${chosenProject.namespace}/${chosenProject.name}`)

      // Remember this choice
      await this.saveLastUsedProject(chosenProject.path)

      return chosenProject.path
    }

    // No valid projects found, create a new one
    console.log('No valid projects found, creating a new project...')
    const newProjectPath = await this.createDefaultProject(cliPath)
    await this.saveLastUsedProject(newProjectPath)
    return newProjectPath
  }

  /**
   * Get last used project path
   */
  private async getLastUsedProject(): Promise<string | null> {
    try {
      const content = await fsPromises.readFile(this.lastProjectFile, 'utf-8')
      return content.trim()
    } catch {
      return null
    }
  }

  /**
   * Save last used project path
   */
  private async saveLastUsedProject(projectPath: string): Promise<void> {
    try {
      await fsPromises.writeFile(this.lastProjectFile, projectPath, 'utf-8')
    } catch (error) {
      console.error('Failed to save last project:', error)
    }
  }

  /**
   * Check if a project has a valid configuration
   */
  private async isValidProject(projectPath: string): Promise<boolean> {
    try {
      const configPath = path.join(projectPath, 'llamafarm.yaml')
      const content = await fsPromises.readFile(configPath, 'utf-8')

      // Basic validation: must have version, name, namespace, and runtime
      const hasVersion = content.includes('version:')
      const hasName = content.includes('name:')
      const hasNamespace = content.includes('namespace:')
      const hasRuntime = content.includes('runtime:')
      const isNotTooSmall = content.length > 200

      return hasVersion && hasName && hasNamespace && hasRuntime && isNotTooSmall
    } catch {
      return false
    }
  }

  /**
   * Create a default project
   */
  private async createDefaultProject(cliPath: string): Promise<string> {
    const timestamp = Date.now()
    const projectName = `desktop-${timestamp}`
    const namespace = 'default'
    const projectPath = path.join(this.defaultProjectsDir, namespace, projectName)

    try {
      // Ensure parent directory exists
      await fsPromises.mkdir(path.join(this.defaultProjectsDir, namespace), { recursive: true })

      // Run lf init
      console.log(`Creating new project: ${namespace}/${projectName}`)
      await execAsync(`"${cliPath}" init ${projectName} --namespace ${namespace}`, {
        cwd: path.join(this.defaultProjectsDir, namespace)
      })

      console.log(`✓ Project created at: ${projectPath}`)
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
