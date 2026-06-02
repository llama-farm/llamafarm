/** Base error for all LlamaFarm SDK errors. */
export class LlamaFarmError extends Error {
  readonly statusCode?: number
  readonly detail?: string

  constructor(message: string, statusCode?: number, detail?: string) {
    super(message)
    this.name = 'LlamaFarmError'
    this.statusCode = statusCode
    this.detail = detail
  }
}

/** Cannot connect to LlamaFarm server. */
export class ConnectionError extends LlamaFarmError {
  constructor(message: string) {
    super(message)
    this.name = 'ConnectionError'
  }
}

/** Resource not found (project, model, job, etc.). */
export class NotFoundError extends LlamaFarmError {
  constructor(message: string, detail?: string) {
    super(message, 404, detail)
    this.name = 'NotFoundError'
  }
}

/** Request validation failed. */
export class ValidationError extends LlamaFarmError {
  constructor(message: string, detail?: string) {
    super(message, 422, detail)
    this.name = 'ValidationError'
  }
}

/** Server-side error. */
export class ServerError extends LlamaFarmError {
  constructor(message: string, statusCode?: number, detail?: string) {
    super(message, statusCode ?? 500, detail)
    this.name = 'ServerError'
  }
}
