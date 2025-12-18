import { useQuery } from '@tanstack/react-query'

interface GitHubRepoResponse {
  stargazers_count: number
  full_name: string
  html_url: string
}

/**
 * Hook to fetch GitHub stars count for the llamafarm repository
 * @returns Query result with stars count
 */
export function useGitHubStars() {
  return useQuery<GitHubRepoResponse>({
    queryKey: ['github', 'stars', 'llama-farm', 'llamafarm'],
    queryFn: async () => {
      const response = await fetch(
        'https://api.github.com/repos/llama-farm/llamafarm'
      )
      if (!response.ok) {
        throw new Error('Failed to fetch GitHub stars')
      }
      return response.json()
    },
    staleTime: 5 * 60 * 1000, // Consider data fresh for 5 minutes
    refetchInterval: 10 * 60 * 1000, // Refetch every 10 minutes to stay up to date
    retry: 2,
    refetchOnWindowFocus: true, // Refetch when window regains focus
  })
}



