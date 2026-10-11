// Downsamples the microphone to 16 kHz mono PCM16 and posts 100 ms chunks to the main thread.
// Each output sample is the average of the input samples it covers, a cheap low-pass that
// keeps aliasing down compared with picking samples.
class PcmRecorder extends AudioWorkletProcessor {
  constructor() {
    super()
    this.ratio = sampleRate / 16000
    this.sum = 0
    this.count = 0
    this.acc = 0
    this.chunk = new Int16Array(1600)
    this.filled = 0
  }

  process(inputs) {
    const input = inputs[0]?.[0]
    if (!input) return true
    for (let i = 0; i < input.length; i++) {
      this.sum += input[i]
      this.count++
      this.acc++
      if (this.acc >= this.ratio) {
        this.acc -= this.ratio
        this.push(this.sum / this.count)
        this.sum = 0
        this.count = 0
      }
    }
    return true
  }

  push(sample) {
    const s = Math.max(-1, Math.min(1, sample))
    this.chunk[this.filled++] = s < 0 ? s * 0x8000 : s * 0x7fff
    if (this.filled === this.chunk.length) {
      this.port.postMessage(this.chunk.buffer, [this.chunk.buffer])
      this.chunk = new Int16Array(1600)
      this.filled = 0
    }
  }
}

registerProcessor('pcm-recorder', PcmRecorder)
