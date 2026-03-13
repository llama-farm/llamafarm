import { useState, useCallback, useRef, useEffect } from 'react'
import type { VoiceEntry } from '../types'

interface UseVoiceOptions {
  onCommand?: (command: string) => void
}

export function useVoice({ onCommand }: UseVoiceOptions = {}) {
  const [isListening, setIsListening] = useState(false)
  const [lastMessage, setLastMessage] = useState<string>('')
  const [entries, setEntries] = useState<VoiceEntry[]>([])
  const recognitionRef = useRef<any>(null)
  const synthRef = useRef<SpeechSynthesis | null>(null)

  useEffect(() => {
    if (typeof window !== 'undefined') {
      synthRef.current = window.speechSynthesis
    }
  }, [])

  // Add entry to voice log
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

  // Speak text aloud (drone/system voice)
  const speak = useCallback((text: string, speaker: VoiceEntry['speaker'] = 'drone') => {
    addEntry(speaker, text)
    
    if (synthRef.current) {
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1.1
      utterance.pitch = 0.9
      utterance.volume = 0.8
      // Try to pick a neutral/robotic voice
      const voices = synthRef.current.getVoices()
      const preferred = voices.find(v => v.name.includes('Daniel') || v.name.includes('Samantha') || v.name.includes('Google'))
      if (preferred) utterance.voice = preferred
      synthRef.current.speak(utterance)
    }
  }, [addEntry])

  // Start listening
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

    recognition.onend = () => {
      setIsListening(false)
    }

    recognition.onerror = () => {
      setIsListening(false)
    }

    recognitionRef.current = recognition
    recognition.start()
    setIsListening(true)
  }, [addEntry, onCommand])

  // Stop listening
  const stopListening = useCallback(() => {
    recognitionRef.current?.stop()
    setIsListening(false)
  }, [])

  // Toggle
  const toggleListening = useCallback(() => {
    if (isListening) stopListening()
    else startListening()
  }, [isListening, startListening, stopListening])

  // Add a detection callout
  const announceDetection = useCallback((type: string, confidence: number, mgrs: string, detectionId: string) => {
    const text = `${type} detected — ${confidence}% — grid ${mgrs}`
    addEntry('drone', text, detectionId, type as VoiceEntry['type'])
    
    if (synthRef.current) {
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1.1
      utterance.pitch = 0.9
      utterance.volume = 0.8
      synthRef.current.speak(utterance)
    }
  }, [addEntry])

  return {
    isListening,
    lastMessage,
    entries,
    speak,
    startListening,
    stopListening,
    toggleListening,
    addEntry,
    announceDetection,
  }
}
