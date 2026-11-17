/**
 * ============================================================================
 *  PlateOCR.js — Google ML Kit Text Recognition for the Visual Extraction Layer
 * ============================================================================
 *  Project : Design and Verification of a Regular Language Parser for
 *            MMDA Coding Rules Using Finite State Automata
 *  Course  : COSC 302 – Automata and Language Theory
 *  Group 12: BULAN, ZALDY MARCUS, B.
 *            Del Mundo, M. A.
 *            Librado, C. T.
 *            Paz, L. H.
 *
 *  Description:
 *    Primary OCR engine. Runs Google ML Kit text recognition on the phone
 *    (offline, free, bundled model) through the local Expo module in
 *    modules/plate-text-recognizer. Camera shots are first cropped to the
 *    on-screen scan frame with expo-image-manipulator so other text on the
 *    car (brand badges, stickers) is left out.
 *
 *    ML Kit is a native module: it works in the APK build, not in Expo Go.
 *    In Expo Go it is missing, so the app asks the user to type the plate.
 * ============================================================================
 */

import { Platform } from 'react-native';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { isAvailable as nativeAvailable, recognizeLines } from './modules/plate-text-recognizer';

export const ML_KIT_AVAILABLE = Platform.OS !== 'web' && nativeAvailable;

/**
 * Maps the scan frame (drawn over a "cover"-scaled camera preview) back into
 * photo pixels, with padding in case the preview and photo framing differ.
 *
 * @param {number} photoW  photo width in pixels
 * @param {number} photoH  photo height in pixels
 * @param {{boxW:number, boxH:number, x:number, y:number, w:number, h:number, pad?:number}} crop
 */
export function frameToPhotoRect(photoW, photoH, crop) {
  const s = Math.max(crop.boxW / photoW, crop.boxH / photoH);
  const offX = (crop.boxW - photoW * s) / 2;
  const offY = (crop.boxH - photoH * s) / 2;
  const x = (crop.x - offX) / s;
  const y = (crop.y - offY) / s;
  const w = crop.w / s;
  const h = crop.h / s;
  const pad = crop.pad ?? 0.15;
  const padX = w * pad;
  const padY = h * pad * 1.6;
  const originX = Math.max(0, Math.round(x - padX));
  const originY = Math.max(0, Math.round(y - padY));
  const endX = Math.min(photoW, Math.round(x + w + padX));
  const endY = Math.min(photoH, Math.round(y + h + padY));
  if (endX - originX < 40 || endY - originY < 40) return null;
  return { originX, originY, width: endX - originX, height: endY - originY };
}

async function extract(uri) {
  const lines = await recognizeLines(uri);
  return (lines || []).join('\n');
}

/**
 * Reads text from a photo with ML Kit.
 *
 * @param {string} uri  local photo URI
 * @param {{ width?: number, height?: number, crop?: object|null }} [opts]
 * @returns {Promise<{ text: string, fullText: () => Promise<string> }>}
 *   text      – text read inside the scan frame (or the whole photo if no crop)
 *   fullText  – reads the whole, uncropped photo (used as a fallback)
 */
export async function recognizeWithMlKit(uri, { width, height, crop } = {}) {
  if (!ML_KIT_AVAILABLE) throw new Error('ML Kit is not available in this build.');
  const fullText = () => extract(uri);

  const rect = crop && width && height ? frameToPhotoRect(width, height, crop) : null;
  if (!rect) return { text: await fullText(), fullText };

  const ref = await ImageManipulator.manipulate(uri).crop(rect).renderAsync();
  const out = await ref.saveAsync({ format: SaveFormat.JPEG, compress: 0.95 });
  try {
    ref.release?.(); // free the decoded photo right away
  } catch {
    /* ignore */
  }
  return { text: await extract(out.uri), fullText };
}
