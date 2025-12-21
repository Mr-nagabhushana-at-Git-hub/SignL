import { GoogleGenAI } from "@google/genai";
import { ConvoEntry } from "../types";

// Note: In a real environment, this key comes from process.env.API_KEY
// The user must ensure the environment is set up correctly.
const API_KEY = process.env.API_KEY || "";

const ai = new GoogleGenAI({ apiKey: API_KEY });

export const generateSummary = async (logs: ConvoEntry[]): Promise<string> => {
  if (!API_KEY) return "API Key missing. Cannot generate summary.";
  
  const transcript = logs
    .filter(l => l.type !== 'system')
    .map(l => `[${l.timestamp}] ${l.speaker.name} (${l.type}): ${l.text}`)
    .join('\n');

  if (transcript.length < 10) return "Not enough conversation data to summarize.";

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: `
        You are an advanced accessibility assistant for SignSync Omni.
        Summarize the following conversation transcript between a Deaf user (using Sign Language) and a Hearing user (using Voice).
        Focus on the key action items and emotional tone. Keep it concise (under 50 words).
        
        Transcript:
        ${transcript}
      `,
    });

    return response.text || "No summary generated.";
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "Error connecting to Neural Engine.";
  }
};