/**
 * ============================================================================
 *  Project : Design and Verification of a Regular Language Parser for
 *            MMDA Coding Rules Using Finite State Automata
 *  Course  : COSC 302 – Automata and Language Theory
 *  Group 12
 * ============================================================================
 */

/*
 *  plate-text-recognizer — JS side of the local ML Kit module.
 *  Returns null for the native module in Expo Go (it isn't compiled there),
 *  so the app can show a message instead of crashing.
 */
import { requireOptionalNativeModule } from 'expo';

const Native = requireOptionalNativeModule('PlateTextRecognizer');

export const isAvailable = Native != null;

/**
 * Runs ML Kit text recognition on a local image.
 * @param {string} uri  file:// or content:// URI
 * @returns {Promise<string[]>} recognized lines, top to bottom
 */
export function recognizeLines(uri) {
  if (!Native) return Promise.reject(new Error('ML Kit is not available in this build.'));
  return Native.recognize(uri);
}
