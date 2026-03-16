import { useState, useCallback, useRef, useEffect } from 'react'
import type { VoiceEntry } from '../types'

interface UseVoiceOptions {
  onCommand?: (command: string) => void
}

function loadVolume(): number {
  try {
    const v = localStorage.getItem('overwatch-volume')
    return v !== null ? parseFloat(v) : 0.8
  } catch { return 0.8 }
}

function loadMuted(): boolean {
  try { return localStorage.getItem('overwatch-muted') === 'true' } catch { return false }
}

export function useVoice({ onCommand }: UseVoiceOptions = {}) {
  const [isListening, setIsListening] = useState(false)
  const [lastMessage, setLastMessage] = useState<string>('')
  const [entries, setEntries] = useState<VoiceEntry[]>([])
  const [volume, setVolumeState] = useState(loadVolume)
  const [muted, setMutedState] = useState(loadMuted)
  const recognitionRef = useRef<any>(null)
  const synthRef = useRef<SpeechSynthesis | null>(null)

  useEffect(() => {
    if (typeof window !== 'undefined') {
      synthRef.current = window.speechSynthesis
    }
  }, [])

  const setVolume = useCallback((v: number) => {
    setVolumeState(v)
    try { localStorage.setItem('overwatch-volume', String(v)) } catch {}
    if (v > 0) {
      setMutedState(false)
      try { localStorage.setItem('overwatch-muted', 'false') } catch {}
    }
  }, [])

  const toggleMute = useCallback(() => {
    setMutedState(prev => {
      const next = !prev
      try { localStorage.setItem('overwatch-muted', String(next)) } catch {}
      return next
    })
  }, [])

  const addEntry = useCallback((speaker: VoiceEntry['speaker'], text: string, detectionId?: string, type?: VoiceEntry['type']) => {
    const entry: VoiceEntry = {
      id: `voice-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      speaker,
      text,
      timestamp: new Date(),
      detectionId,
      type,
    }
    setEntries(prev => [...prev, entry])
    setLastMessage(text)
    return entry
  }, [])

  const speak = useCallback((text: string, speaker: VoiceEntry['speaker'] = 'drone') => {
    addEntry(speaker, text)

    if (synthRef.current && !muted && volume > 0) {
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1.1
      utterance.pitch = 0.9
      utterance.volume = volume
      const voices = synthRef.current.getVoices()
      const preferred = voices.find(v => v.name.includes('Daniel') || v.name.includes('Samantha') || v.name.includes('Google'))
      if (preferred) utterance.voice = preferred
      synthRef.current.speak(utterance)
    }
  }, [addEntry, muted, volume])

  const startListening = useCallback(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRecognition) {
      console.warn('Speech recognition not supported')
      return
    }

    const recognition = new SpeechRecognition()
    recognition.continuous = false
    recognition.interimResults = false
    recognition.lang = 'en-US'

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript.toLowerCase().trim()
      addEntry('operator', transcript)
      onCommand?.(transcript)
    }

    recognition.onend = () => { setIsListening(false) }
    recognition.onerror = () => { setIsListening(false) }

    recognitionRef.current = recognition
    recognition.start()
    setIsListening(true)
  }, [addEntry, onCommand])

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop()
    setIsListening(false)
  }, [])

  const toggleListening = useCallback(() => {
    if (isListening) stopListening()
    else startListening()
  }, [isListening, startListening, stopListening])

  const announceDetection = useCallback((type: string, confidence: number, mgrs: string, detectionId: string) => {
    const text = `${type} detected — ${confidence}% — grid ${mgrs}`
    addEntry('drone', text, detectionId, type as VoiceEntry['type'])

    if (synthRef.current && !muted && volume > 0) {
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1.1
      utterance.pitch = 0.9
      utterance.volume = volume
      synthRef.current.speak(utterance)
    }
  }, [addEntry, muted, volume])

  return {
    isListening,
    lastMessage,
    entries,
    volume,
    muted,
    speak,
    setVolume,
    toggleMute,
    startListening,
    stopListening,
    toggleListening,
    addEntry,
    announceDetection,
  }
}
