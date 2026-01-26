/**
 * AudioWorklet processor for capturing raw PCM audio samples.
 *
 * Captures audio from the microphone, converts Float32 samples to Int16 PCM,
 * and sends chunks to the main thread for WebSocket transmission.
 *
 * Output format: 16kHz, 16-bit signed integer, mono (little-endian)
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    // Buffer size: 2048 samples = ~128ms at 16kHz
    // This provides a good balance between latency and efficiency
    this.bufferSize = 2048
    this.buffer = new Float32Array(this.bufferSize)
    this.bufferIndex = 0
  }

  /**
   * Process audio samples from the input.
   * Called by the audio rendering thread with 128 samples per call at a time.
   */
  process(inputs) {
    const input = inputs[0]
    if (!input || !input[0]) {
      return true // Keep processor alive
    }

    const channelData = input[0] // Mono channel

    // Accumulate samples into our buffer
    for (let i = 0; i < channelData.length; i++) {
      this.buffer[this.bufferIndex++] = channelData[i]

      // When buffer is full, convert and send
      if (this.bufferIndex >= this.bufferSize) {
        // Convert Float32 (-1.0 to 1.0) to Int16 (-32768 to 32767)
        const pcm = new Int16Array(this.bufferSize)
        for (let j = 0; j < this.bufferSize; j++) {
          // Clamp and scale
          const sample = Math.max(-1, Math.min(1, this.buffer[j]))
          pcm[j] = sample < 0 ? sample * 32768 : sample * 32767
        }

        // Send PCM buffer to main thread (transfer ownership for efficiency)
        this.port.postMessage(pcm.buffer, [pcm.buffer])

        // Reset buffer
        this.buffer = new Float32Array(this.bufferSize)
        this.bufferIndex = 0
      }
    }

    return true // Keep processor alive
  }
}

registerProcessor('pcm-processor', PCMProcessor)
