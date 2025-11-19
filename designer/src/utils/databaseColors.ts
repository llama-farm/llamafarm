/**
 * Database color utility
 * Assigns consistent colors to databases based on their order in the project config
 */

interface Database {
  name: string
}

/**
 * Get the color classes for a database badge based on its position in the databases array
 * Colors cycle through 6 options: teal, sky, indigo, purple, fuchsia, emerald
 *
 * @param databaseName - The name of the database
 * @param databases - Array of all databases from project config
 * @returns Tailwind CSS classes for background and text color
 */
export function getDatabaseColor(
  databaseName: string,
  databases: Database[]
): string {
  // Find the index of this database in the list
  const index = databases.findIndex(db => db.name === databaseName)

  // If database not found, return default teal color
  if (index === -1) {
    return 'bg-teal-300 text-white dark:bg-teal-500 dark:text-slate-900'
  }

  // Cycle through 6 colors using modulo
  const colorIndex = index % 6

  // Map index to color classes
  const colors = [
    'bg-teal-300 text-slate-900 dark:bg-teal-400 dark:text-slate-900', // 0: teal
    'bg-sky-300 text-slate-900 dark:bg-sky-400 dark:text-slate-900', // 1: sky
    'bg-indigo-300 text-slate-900 dark:bg-indigo-400 dark:text-slate-900', // 2: indigo
    'bg-purple-300 text-slate-900 dark:bg-purple-400 dark:text-slate-900', // 3: purple
    'bg-fuchsia-300 text-slate-900 dark:bg-fuchsia-400 dark:text-slate-900', // 4: fuchsia
    'bg-emerald-300 text-slate-900 dark:bg-emerald-400 dark:text-slate-900', // 5: emerald
  ]

  return colors[colorIndex]
}
