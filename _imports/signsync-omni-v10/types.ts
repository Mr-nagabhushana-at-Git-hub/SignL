export interface Demographics {
  age: number;
  gender: 'Male' | 'Female' | 'Non-Binary';
  confidence: number;
}

export interface UserIdentity {
  id: string;
  name: string;
  role: 'Admin' | 'Guest' | 'Host';
  isVerified: boolean;
}

export interface ConvoEntry {
  id: string;
  speaker: UserIdentity;
  demographics?: Demographics;
  text: string;
  timestamp: string;
  type: 'voice-input' | 'sign-detected' | 'system' | 'ai-summary';
  translationConfidence?: number;
}

export enum AppState {
  IDLE = 'IDLE',
  LISTENING = 'LISTENING',
  WATCHING = 'WATCHING',
  PROCESSING = 'PROCESSING',
  SPEAKING = 'SPEAKING',
}

// Digital Certificate Interface
export interface DigitalCertificate {
  owner: string;
  email: string;
  productVersion: string;
  encryptionKey: string;
  issuedAt: string;
}