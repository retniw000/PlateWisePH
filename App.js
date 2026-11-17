/**
 * ============================================================================
 *  App.js — PlateWise PH: Visual Parser UI and Preprocessing Layer
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
 *    1. Visual Extraction Layer  – live camera (expo-camera) or gallery photo
 *                                  → Google ML Kit OCR (PlateOCR.js), or
 *                                  manual plate entry.
 *    2. String Standardization   – strips whitespace, noise tokens and
 *                                  formatting anomalies into a clean
 *                                  character array (standardizeInput).
 *    3. Calendar + Time symbols  – converts the day into one symbol
 *                                  (Mon … Sun, Hol) and the time into 'p'/'o'.
 *    4. Automata Acceptor        – hands the symbol array to DFAEngine.js and
 *                                  shows only the final verdict (the DFA runs
 *                                  behind the scenes; no trace is displayed).
 *
 *  Screens: Splash → Onboarding → Check plate / Scan plate / Coding result,
 *           Schedule tab, Guide tab. Saved plates are kept on the phone
 *           (AsyncStorage) and offered on Check plate and Schedule.
 * ============================================================================
 */

import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Alert,
  Animated,
  BackHandler,
  Easing,
  Image,
  LayoutAnimation,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
  useColorScheme,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider, SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import * as SplashScreen from 'expo-splash-screen';
import { useFonts } from 'expo-font';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Svg, { Path } from 'react-native-svg';
import { Feather, MaterialCommunityIcons } from '@expo/vector-icons';

import { ML_KIT_AVAILABLE, recognizeWithMlKit } from './PlateOCR';
import { DAY_SYMBOLS, GROUP_CODING_DAY, VERDICT, runDFA } from './DFAEngine';

SplashScreen.preventAutoHideAsync().catch(() => {});

// ============================================================================
//  PREPROCESSING LAYER
// ============================================================================

/** Text commonly printed on Philippine plates that is not part of the plate number. */
export const NOISE_TOKENS = [
  'REPUBLIC OF THE PHILIPPINES',
  'REPUBLIKA NG PILIPINAS',
  'LAND TRANSPORTATION OFFICE',
  'METRO MANILA',
  'PHILIPPINES',
  'PILIPINAS',
  'FOR HIRE',
  'GOVERNMENT',
  'DIPLOMATIC',
  'MINDANAO',
  'VISAYAS',
  'PRIVATE',
  'LUZON',
  'NCR',
  'LTO',
];

/** OCR look-alike fixes, applied by position (letters at 1–3, digits at 4–7). OCR input only. */
const DIGIT_TO_LETTER = { 0: 'O', 1: 'I', 2: 'Z', 4: 'A', 5: 'S', 6: 'G', 7: 'T', 8: 'B' };
const LETTER_TO_DIGIT = { O: '0', Q: '0', D: '0', U: '0', I: '1', L: '1', Z: '2', A: '4', S: '5', G: '6', T: '7', B: '8' };

const PLATE_PATTERN = /[A-Z]{3}[0-9]{4}/;
/** OCR swaps that happen constantly on plate fonts; they cost less when ranking candidates. */
const COMMON_SWAPS = new Set(['O→0', 'Q→0', 'D→0', 'I→1', 'L→1', '0→O', '1→I']);
const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function stripNoise(text) {
  let out = text;
  const removed = [];
  [...NOISE_TOKENS]
    .sort((a, b) => b.length - a.length)
    .forEach((token) => {
      const re = new RegExp(`\\b${escapeRegex(token).replace(/ /g, '\\s+')}\\b`, 'g');
      if (re.test(out)) {
        removed.push(token);
        out = out.replace(re, ' ');
      }
    });
  return { text: out, removed };
}

const collapse = (text) => text.replace(/[^A-Z0-9]/g, '');

function correctLookalikes(seven) {
  const fixes = [];
  const chars = seven.split('').map((c, i) => {
    if (i < 3 && DIGIT_TO_LETTER[c]) {
      fixes.push(`${c}→${DIGIT_TO_LETTER[c]} (pos ${i + 1})`);
      return DIGIT_TO_LETTER[c];
    }
    if (i >= 3 && LETTER_TO_DIGIT[c]) {
      fixes.push(`${c}→${LETTER_TO_DIGIT[c]} (pos ${i + 1})`);
      return LETTER_TO_DIGIT[c];
    }
    return c;
  });
  return { value: chars.join(''), fixes };
}

/**
 * Every plate-shaped reading (L L L D D D D) hidden in OCR text, best first.
 * Slides a 7-character window over each line (and over pairs of adjacent lines),
 * applies the positional look-alike fixes, and scores each result:
 * cost = look-alike fixes (0.5 for very common swaps like I↔1, O↔0; 1 for the rest)
 *        + 0.6 per stray character on that line.
 *
 * @param {string} raw  OCR text
 * @returns {{ plate: string, cost: number }[]}
 */
export function plateCandidates(raw) {
  if (!raw || !raw.trim()) return [];
  const lines = raw.normalize('NFKC').toUpperCase().split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const pieces = new Set();
  lines.forEach((line, i) => {
    const clean = collapse(stripNoise(line).text);
    pieces.add(collapse(line));
    pieces.add(clean);
    if (i < lines.length - 1) pieces.add(clean + collapse(stripNoise(lines[i + 1]).text));
  });
  const best = new Map();
  pieces.forEach((piece) => {
    for (let k = 0; k + 7 <= piece.length; k += 1) {
      const { value, fixes } = correctLookalikes(piece.slice(k, k + 7));
      if (!/^[A-Z]{3}[0-9]{4}$/.test(value)) continue;
      const fixCost = fixes.reduce((sum, f) => sum + (COMMON_SWAPS.has(f.slice(0, 3)) ? 0.5 : 1), 0);
      const cost = fixCost + (piece.length - 7) * 0.6;
      if (!best.has(value) || best.get(value) > cost) best.set(value, cost);
    }
  });
  return [...best.entries()].sort((a, b) => a[1] - b[1]).map(([plate, cost]) => ({ plate, cost }));
}

/**
 * Majority vote across several readings of the same plate, character by character.
 * Ties go to the earliest reading.
 *
 * @param {(string|null)[]} readings  best plate from each shot (null if none)
 * @returns {{ plate: string, agree: number, valid: number, total: number } | null}
 */
export function votePlate(readings) {
  const valid = readings.filter(Boolean);
  if (!valid.length) return null;
  let plate = '';
  for (let i = 0; i < 7; i += 1) {
    const counts = new Map();
    valid.forEach((p) => counts.set(p[i], (counts.get(p[i]) || 0) + 1));
    plate += [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
  }
  return { plate, agree: valid.filter((p) => p === plate).length, valid: valid.length, total: readings.length };
}

/**
 * String Standardization.
 * Takes raw OCR text or a manually typed plate and returns a clean
 * character array for the DFA. It never decides validity — the DFA does.
 *
 * @param {string} raw
 * @param {'manual'|'ocr'} source
 * @returns {{ plate: string, chars: string[], log: string[] }}
 */
export function standardizeInput(raw, source = 'manual') {
  const log = [];
  if (!raw || !raw.trim()) return { plate: '', chars: [], log };

  // 1. Unicode + case normalization (full-width chars, lowercase, etc.)
  const upper = raw.normalize('NFKC').toUpperCase();
  if (upper !== raw) log.push('Normalized to uppercase / standard characters');

  if (source === 'manual') {
    // Manual entry: strip separators only. No noise-token removal and no look-alike
    // guessing — whatever the user typed reaches the DFA so malformed plates are rejected.
    const plate = collapse(upper);
    if (/[^A-Z0-9]/.test(upper)) log.push('Stripped whitespace, hyphens and symbols');
    return { plate, chars: plate.split(''), log };
  }

  // OCR entry: examine line by line, since a plate photo usually has several text lines.
  const lines = upper.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  log.push(`OCR returned ${lines.length} line(s)`);

  // a0) A line that is exactly a plate before any noise removal (protects plates like "NCR 1234")
  for (const line of lines) {
    const c = collapse(line);
    if (c.length === 7 && PLATE_PATTERN.test(c)) {
      log.push(`Stripped whitespace, hyphens and symbols from "${line}"`);
      return { plate: c, chars: c.split(''), log };
    }
  }

  const removedAll = new Set();
  const cleanedLines = lines.map((line) => {
    const { text, removed } = stripNoise(line);
    removed.forEach((r) => removedAll.add(r));
    return collapse(text);
  });
  if (removedAll.size) log.push(`Removed noise tokens: ${[...removedAll].join(', ')}`);
  log.push('Stripped whitespace, hyphens and symbols');

  // a) A line that already contains L L L D D D D
  for (const line of cleanedLines) {
    const m = line.match(PLATE_PATTERN);
    if (m) {
      if (line.length !== 7) log.push(`Isolated plate segment "${m[0]}" from "${line}"`);
      return { plate: m[0], chars: m[0].split(''), log };
    }
  }

  // b) OCR look-alike errors and stray characters (e.g. "N4B 12O4", "INCA 1231.")
  const cands = plateCandidates(cleanedLines.join('\n'));
  if (cands.length) {
    log.push(`Corrected OCR look-alikes / stray characters → "${cands[0].plate}"`);
    return { plate: cands[0].plate, chars: cands[0].plate.split(''), log };
  }

  // c) Plate split across lines (e.g. "ABC" / "1234")
  const joined = cleanedLines.join('');
  const m = joined.match(PLATE_PATTERN);
  if (m) {
    log.push(`Joined OCR lines and isolated "${m[0]}"`);
    return { plate: m[0], chars: m[0].split(''), log };
  }

  // d) Nothing plate-shaped: pass the longest line through; the DFA will show where it fails.
  const fallback = cleanedLines.reduce((a, b) => (b.length > a.length ? b : a), '');
  log.push('No L-L-L-D-D-D-D segment found; passing longest line to the DFA as-is');
  return { plate: fallback, chars: fallback.split(''), log };
}

// ---------------------------------------------------------------------------
//  Calendar + time symbols
// ---------------------------------------------------------------------------
const JS_DAY_TO_SYMBOL = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

/** Peak windows in minutes from midnight, inclusive: 7:00–10:00 a.m. and 5:00–8:00 p.m. */
export const PEAK_WINDOWS = [
  { start: 7 * 60, end: 10 * 60 },
  { start: 17 * 60, end: 20 * 60 },
];

export function timeToSymbol(hours24, minutes) {
  const t = hours24 * 60 + minutes;
  return PEAK_WINDOWS.some((w) => t >= w.start && t <= w.end) ? 'p' : 'o';
}

const pad = (n) => String(n).padStart(2, '0');
const to24 = (h12, mer) => (h12 % 12) + (mer === 'PM' ? 12 : 0);
const label12 = (h24, m) => `${h24 % 12 === 0 ? 12 : h24 % 12}:${pad(m)} ${h24 >= 12 ? 'PM' : 'AM'}`;

const DAY_NAME = {
  Mon: 'Monday', Tue: 'Tuesday', Wed: 'Wednesday', Thu: 'Thursday',
  Fri: 'Friday', Sat: 'Saturday', Sun: 'Sunday', Hol: 'Holiday',
};
const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];
const WEEK = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const DAY_ENDINGS = { Mon: '1 · 2', Tue: '3 · 4', Wed: '5 · 6', Thu: '7 · 8', Fri: '9 · 0' };

const NCR_CITIES = [
  'Caloocan', 'Las Piñas', 'Makati', 'Malabon', 'Mandaluyong', 'Manila', 'Marikina', 'Muntinlupa',
  'Navotas', 'Parañaque', 'Pasay', 'Pasig', 'Pateros', 'Quezon City', 'San Juan', 'Taguig', 'Valenzuela',
];

const formatPlate = (p) => (/^[A-Z]{3}[0-9]{4}$/.test(p) ? `${p.slice(0, 3)} ${p.slice(3)}` : p);

/** Runs only the plate part of M (S0 … S6 → G) and returns the group state, or null. */
function plateGroup(chars) {
  if (!chars.length) return null;
  const s = runDFA(chars).finalState;
  return /^G[1-5]$/.test(s) ? s : null;
}

/** Full pipeline: standardize → build symbol string w → run M. */
function buildCheck({ raw, source, day, h24, m, holiday, mode, city }) {
  const std = standardizeInput(raw, source);
  const daySymbol = holiday ? 'Hol' : day;
  const timeSymbol = timeToSymbol(h24, m);
  const symbols = std.chars.length ? [...std.chars, daySymbol, timeSymbol] : [];
  const result = runDFA(symbols);
  const valid = result.verdict !== VERDICT.REJECTED;
  return {
    std,
    symbols,
    result,
    valid,
    plateDisplay: formatPlate(std.plate) || '—',
    lastDigit: valid ? std.plate[6] : '—',
    day,
    daySymbol,
    holiday,
    timeLabel: label12(h24, m),
    timeSymbol,
    mode,
    city,
  };
}

/**
 * Coding status of one car at a given moment (runs the same DFA check).
 * kind: 'coding' (restricted now) | 'codingDay' (coding day, outside the hours)
 *       | 'free' (not its coding day) | 'holiday' | 'invalid'
 */
function carStatus(plate, now, holiday) {
  const h24 = now.getHours();
  const m = now.getMinutes();
  const day = JS_DAY_TO_SYMBOL[now.getDay()];
  const c = buildCheck({ raw: plate, source: 'manual', day, h24, m, holiday, mode: 'live', city: '' });
  const codingDay = c.result.codingDay;
  const t = h24 * 60 + m;
  if (!c.valid) return { kind: 'invalid', label: 'Invalid plate', sub: 'Edit or remove this car.', check: c };
  if (c.result.verdict === VERDICT.CODING) {
    const until = t <= 10 * 60 ? '10:00 AM' : '8:00 PM';
    return { kind: 'coding', label: 'In coding now', sub: `Restricted until ${until}`, check: c };
  }
  if (c.result.path.includes('SR')) {
    let sub = 'Coding is over for today';
    if (t < 7 * 60) sub = 'Coding starts at 7:00 AM';
    else if (t < 17 * 60) sub = 'Coding again 5:00–8:00 PM';
    return { kind: 'codingDay', label: 'Coding day', sub, check: c };
  }
  if (holiday) {
    return { kind: 'holiday', label: 'No coding', sub: `Holiday · usual coding day ${DAY_NAME[codingDay]}`, check: c };
  }
  return { kind: 'free', label: 'No coding today', sub: `Coding day: ${DAY_NAME[codingDay]}`, check: c };
}

// ============================================================================
//  THEME
// ============================================================================
const LIGHT = {
  bg: '#F5F7FA',
  card: '#FFFFFF',
  cardLine: '#EDF1F5',
  surfaceAlt: '#F3F6F9',
  rowLine: '#F0F3F6',
  navy: '#152A45',
  navyDeep: '#0F2238',
  ink: '#14263D',
  body: '#3D4B5C',
  muted: '#6B7A8C',
  faint: '#93A0AF',
  placeholder: '#B5BFCB',
  line: '#E3E8EE',
  switchOff: '#CBD3DC',
  backdrop: 'rgba(15,34,56,0.35)',
  teal: '#0F6E73',
  tealBright: '#078A94',
  tealSoft: '#DDEFEF',
  tealLine: '#2F8C8C',
  scan: '#7CCFCB',
  redBg: '#FCEAE5',
  red: '#A8361C',
  amberBg: '#FFF4DC',
  amber: '#8A5A00',
  greenBg: '#E2F3EC',
  green: '#16674A',
  greyBg: '#ECEFF3',
  infoBg: '#E7EDF3',
  nav: '#152A45',
  navMuted: '#93A3B5',
};

const DARK = {
  bg: '#0B1420',
  card: '#142131',
  cardLine: '#1E2D40',
  surfaceAlt: '#1C2A3C',
  rowLine: '#1E2D40',
  navy: '#1A3050',
  navyDeep: '#0A1626',
  ink: '#E8EEF5',
  body: '#C3CEDA',
  muted: '#8E9CAD',
  faint: '#66768A',
  placeholder: '#5A6A7E',
  line: '#26354A',
  switchOff: '#3A4A5E',
  backdrop: 'rgba(0,0,0,0.55)',
  teal: '#2A9E9A',
  tealBright: '#3BC3BE',
  tealSoft: '#163A3E',
  tealLine: '#2F8C8C',
  scan: '#7CCFCB',
  redBg: '#3A1F1B',
  red: '#F2917A',
  amberBg: '#36290F',
  amber: '#EDBE62',
  greenBg: '#13312A',
  green: '#6FD1A6',
  greyBg: '#1F2B3A',
  infoBg: '#18263A',
  nav: '#16283F',
  navMuted: '#8597AD',
};

/** Live palette. Mutated in place by applyTheme so every screen reads the current colors. */
const C = { ...LIGHT };
let currentScheme = 'light';
function applyTheme(scheme) {
  if (scheme === currentScheme) return;
  currentScheme = scheme;
  Object.assign(C, scheme === 'dark' ? DARK : LIGHT);
  styles = makeStyles();
}

const THEME_KEY = 'platewise.theme.v1'; // 'system' | 'light' | 'dark'
const ThemeContext = createContext({ mode: 'system', scheme: 'light', setMode: () => {} });
const useTheme = () => useContext(ThemeContext);

const F = {
  r: 'Inter_400Regular',
  m: 'Inter_500Medium',
  s: 'Inter_600SemiBold',
  b: 'Inter_700Bold',
  x: 'Inter_800ExtraBold',
};

const MONO = Platform.select({ ios: 'Menlo', default: 'monospace' });

// ---------------------------------------------------------------------------
//  Crash note: if the app closes during a native step, the next launch says
//  which step it was (native crashes can't be caught in JavaScript).
// ---------------------------------------------------------------------------
const CRASH_KEY = 'platewise.lastStep.v1';
const SAVED_KEY = 'platewise.savedPlates.v1';
const LAST_PLATE_KEY = 'platewise.lastPlate.v1';
const CARS_KEY = 'platewise.cars.v1';
const MAX_SAVED = 12;
const MAX_NAME = 24;
async function markStep(step) {
  try {
    await AsyncStorage.setItem(CRASH_KEY, step);
  } catch {
    /* best effort */
  }
}
async function clearStep() {
  try {
    await AsyncStorage.removeItem(CRASH_KEY);
  } catch {
    /* best effort */
  }
}
const LOGO_MARK = require('./assets/logo-mark.png');
const LOGO_FULL = require('./assets/logo-full.png');

// ============================================================================
//  SHARED UI
// ============================================================================
function T({ style, w = 'r', children, ...rest }) {
  return (
    <Text style={[{ fontFamily: F[w], color: C.ink }, style]} {...rest}>
      {children}
    </Text>
  );
}

function Brand({ size = 34 }) {
  return (
    <View style={styles.brand}>
      <Image source={LOGO_MARK} style={{ width: size, height: size * 0.79 }} resizeMode="contain" />
      <T w="b" style={{ fontSize: size * 0.68, marginLeft: 8, letterSpacing: -0.3 }}>
        PlateWise
      </T>
      <T w="b" style={{ fontSize: size * 0.26, color: C.teal, marginLeft: 3, marginTop: -size * 0.3 }}>
        PH
      </T>
    </View>
  );
}

function PrimaryButton({ label, onPress, disabled, arrow = true, style }) {
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={({ pressed }) => [styles.primaryBtn, disabled && { opacity: 0.45 }, pressed && { opacity: 0.88 }, style]}
    >
      <T w="s" style={styles.primaryBtnText}>
        {label}
      </T>
      {arrow && <Feather name="arrow-right" size={22} color="#fff" style={styles.primaryBtnArrow} />}
    </Pressable>
  );
}

function ScreenHeader({ title, onBack }) {
  return (
    <View style={styles.screenHeader}>
      <Pressable onPress={onBack} hitSlop={12} style={{ paddingRight: 14 }}>
        <Feather name="arrow-left" size={24} color={C.ink} />
      </Pressable>
      <T w="s" style={{ fontSize: 20 }}>
        {title}
      </T>
    </View>
  );
}

function Card({ style, children }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

function ScanCorners({ color = C.scan, size = 26, thickness = 3 }) {
  const base = { position: 'absolute', width: size, height: size, borderColor: color };
  return (
    <>
      <View style={[base, { top: 0, left: 0, borderTopWidth: thickness, borderLeftWidth: thickness }]} />
      <View style={[base, { top: 0, right: 0, borderTopWidth: thickness, borderRightWidth: thickness }]} />
      <View style={[base, { bottom: 0, left: 0, borderBottomWidth: thickness, borderLeftWidth: thickness }]} />
      <View style={[base, { bottom: 0, right: 0, borderBottomWidth: thickness, borderRightWidth: thickness }]} />
    </>
  );
}

function PlateGraphic({ plate = 'NCA 1231', width = 240 }) {
  return (
    <View style={[styles.plate, { width }]}>
      <T w="s" style={styles.plateTop}>
        PILIPINAS
      </T>
      <T w="x" style={[styles.plateText, { fontSize: width * 0.165 }]}>
        {plate}
      </T>
    </View>
  );
}

function useScanLine(active, distance) {
  const v = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    if (!active) {
      v.stopAnimation();
      v.setValue(0.5);
      return undefined;
    }
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(v, { toValue: 1, duration: 1400, easing: Easing.inOut(Easing.quad), useNativeDriver: true }),
        Animated.timing(v, { toValue: 0, duration: 1400, easing: Easing.inOut(Easing.quad), useNativeDriver: true }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, [active, v]);
  return v.interpolate({ inputRange: [0, 1], outputRange: [-distance / 2, distance / 2] });
}

function Stepper({ value, onMinus, onPlus }) {
  return (
    <View style={styles.stepper}>
      <Pressable onPress={onMinus} style={styles.stepBtn} hitSlop={6}>
        <Feather name="minus" size={18} color={C.ink} />
      </Pressable>
      <T w="s" style={styles.stepValue}>
        {value}
      </T>
      <Pressable onPress={onPlus} style={styles.stepBtn} hitSlop={6}>
        <Feather name="plus" size={18} color={C.ink} />
      </Pressable>
    </View>
  );
}

function Chip({ label, active, onPress }) {
  return (
    <Pressable onPress={onPress} style={[styles.chip, active && styles.chipActive]}>
      <T w="s" style={[styles.chipText, active && { color: '#fff' }]}>
        {label}
      </T>
    </Pressable>
  );
}

function RestrictedHours({ color }) {
  return (
    <View style={styles.hoursRow}>
      <View style={{ flex: 1 }}>
        <T w="s" style={[styles.hoursTime, { color }]}>
          7:00–10:00 AM
        </T>
        <T style={[styles.hoursLabel, { color }]}>Morning</T>
      </View>
      <View style={{ flex: 1 }}>
        <T w="s" style={[styles.hoursTime, { color }]}>
          5:00–8:00 PM
        </T>
        <T style={[styles.hoursLabel, { color }]}>Evening</T>
      </View>
    </View>
  );
}

function WeekGuide({ highlightDay, restricted }) {
  return (
    <View style={styles.weekGuide}>
      {WEEKDAYS.map((d) => {
        const on = d === highlightDay;
        return (
          <View key={d} style={[styles.weekCell, on && { backgroundColor: restricted ? C.redBg : C.tealSoft }]}>
            <T w="s" style={[styles.weekDay, on && { color: restricted ? C.red : C.teal }]}>
              {d.toUpperCase()}
            </T>
            <T w="s" style={[styles.weekEnd, on && { color: restricted ? C.red : C.teal }]}>
              {DAY_ENDINGS[d]}
            </T>
          </View>
        );
      })}
    </View>
  );
}

function InfoNote({ children }) {
  return (
    <View style={styles.infoNote}>
      <Feather name="info" size={18} color={C.muted} style={{ marginTop: 1 }} />
      <T style={styles.infoText}>{children}</T>
    </View>
  );
}

// ============================================================================
//  SPLASH + ONBOARDING
// ============================================================================
function SplashView() {
  return (
    <View style={[styles.fill, { alignItems: 'center', justifyContent: 'center', backgroundColor: C.bg }]}>
      <Image source={LOGO_FULL} style={{ width: 230, height: 246 }} resizeMode="contain" />
    </View>
  );
}

function RoadLines() {
  return (
    <Svg width="100%" height="100%" viewBox="0 0 372 300" preserveAspectRatio="none" style={StyleSheet.absoluteFill}>
      <Path d="M150 300 C 220 240, 300 230, 372 150" stroke="#2A4462" strokeWidth={34} fill="none" opacity={0.55} />
      <Path d="M150 300 C 220 240, 300 230, 372 150" stroke="#5E7896" strokeWidth={2} strokeDasharray="10 10" fill="none" opacity={0.7} />
      <Path d="M0 250 C 60 230, 110 250, 160 300" stroke="#2A4462" strokeWidth={22} fill="none" opacity={0.45} />
    </Svg>
  );
}

function OnboardingScreen({ onStart }) {
  const line = useScanLine(true, 60);
  return (
    <SafeAreaView style={styles.fill} edges={['top', 'bottom']}>
      <ScrollView contentContainerStyle={[styles.page, { flexGrow: 1 }]}>
        <Brand />
        <View style={{ flex: 1, justifyContent: 'center', paddingVertical: 24 }}>
          <View style={[styles.heroCard, styles.heroCenter, { height: 220 }]}>
            <RoadLines />
            <View style={styles.heroFrame}>
              <ScanCorners />
              <PlateGraphic width={220} />
              <Animated.View style={[styles.scanLine, { transform: [{ translateY: line }] }]} />
            </View>
          </View>
          <T w="b" style={[styles.h1, { marginTop: 30 }]}>
            Know your coding.{'\n'}Plan your drive.
          </T>
          <T style={styles.sub}>Scan or type your plate to see when you can drive.</T>
        </View>
        <PrimaryButton label="Get started" onPress={onStart} />
        <T style={[styles.footnote, { textAlign: 'center', marginTop: 14 }]}>Metro Manila · MMDA number coding</T>
      </ScrollView>
    </SafeAreaView>
  );
}

// ============================================================================
//  CHECK PLATE (home)
// ============================================================================
/** Saved cars as tappable chips that fill the plate field. */
function SavedPlates({ cars, current, onPick, title = 'My cars' }) {
  if (!cars.length) return null;
  const cur = collapse((current || '').toUpperCase());
  return (
    <View style={{ marginTop: 16 }}>
      <T w="s" style={styles.tinyLabel}>
        {title.toUpperCase()}
      </T>
      <View style={[styles.wrap, { marginTop: 8 }]}>
        {cars.map((car) => {
          const on = car.plate === cur;
          return (
            <Pressable
              key={car.id}
              onPress={() => onPick(formatPlate(car.plate))}
              style={[styles.savedChip, on && styles.savedChipOn]}
              hitSlop={4}
            >
              <MaterialCommunityIcons name="car-outline" size={16} color={on ? '#fff' : C.teal} />
              <T w="s" style={{ fontSize: 14, marginLeft: 6, color: on ? '#fff' : C.ink }} numberOfLines={1}>
                {car.name}
              </T>
              <T w="m" style={{ fontSize: 12, marginLeft: 6, color: on ? '#D9ECEC' : C.muted }}>
                {formatPlate(car.plate)}
              </T>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

/** Bottom sheet to add a car, save a checked plate under a name, or rename a car. */
function CarEditor({ editor, cars, onClose, onSubmit }) {
  const insets = useSafeAreaInsets();
  const [name, setName] = useState('');
  const [plate, setPlate] = useState('');

  useEffect(() => {
    if (editor) {
      setName(editor.name || '');
      setPlate(editor.plate ? formatPlate(editor.plate) : '');
    }
  }, [editor]);

  if (!editor) return null;
  const plateEditable = editor.mode === 'add';
  const std = standardizeInput(plate, 'manual');
  const validPlate = !!plateGroup(std.chars);
  const duplicate = cars.find((c) => c.plate === std.plate && c.id !== editor.id);
  const trimmed = name.trim();
  const canSave = trimmed.length > 0 && validPlate && !duplicate;
  const titles = { add: 'Add a car', save: 'Save this car', rename: 'Rename car' };

  return (
    <Modal visible transparent animationType="slide" onRequestClose={onClose}>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <Pressable style={styles.sheetBackdrop} onPress={onClose} />
        <View style={[styles.sheet, { paddingBottom: insets.bottom + 16 }]}>
          <View style={styles.sheetHandle} />
          <T w="b" style={{ fontSize: 20 }}>
            {titles[editor.mode]}
          </T>
          <T style={{ fontSize: 13, color: C.muted, marginTop: 4 }}>
            Give it a name you’ll recognize, like “Dad’s Vios” or “Work car”.
          </T>

          <T w="s" style={[styles.tinyLabel, { marginTop: 18 }]}>
            CAR NAME
          </T>
          <TextInput
            value={name}
            onChangeText={(t) => setName(t.slice(0, MAX_NAME))}
            placeholder="My car"
            placeholderTextColor={C.placeholder}
            autoFocus
            maxLength={MAX_NAME}
            returnKeyType="done"
            style={styles.sheetInput}
          />

          <T w="s" style={[styles.tinyLabel, { marginTop: 14 }]}>
            PLATE NUMBER
          </T>
          {plateEditable ? (
            <TextInput
              value={plate}
              onChangeText={(t) => setPlate(t.toUpperCase())}
              placeholder="NCA 1231"
              placeholderTextColor={C.placeholder}
              autoCapitalize="characters"
              autoCorrect={false}
              maxLength={16}
              style={[styles.sheetInput, { fontFamily: F.b, letterSpacing: 0.5 }]}
            />
          ) : (
            <T w="b" style={{ fontSize: 20, marginTop: 6 }}>
              {formatPlate(std.plate)}
            </T>
          )}
          {plateEditable && !!plate.trim() && !validPlate && (
            <T style={[styles.helper, { color: C.red }]}>Use 3 letters and 4 digits, like NCA 1231.</T>
          )}
          {!!duplicate && (
            <T style={[styles.helper, { color: C.red }]}>This plate is already saved as “{duplicate.name}”.</T>
          )}

          <PrimaryButton
            label={editor.mode === 'rename' ? 'Save name' : 'Save car'}
            arrow={false}
            disabled={!canSave}
            onPress={() => onSubmit({ id: editor.id, plate: std.plate, name: trimmed })}
            style={{ marginTop: 20 }}
          />
          <Pressable onPress={onClose} style={{ alignSelf: 'center', marginTop: 14 }} hitSlop={8}>
            <T w="s" style={{ color: C.muted }}>
              Cancel
            </T>
          </Pressable>
        </View>
      </KeyboardAvoidingView>
    </Modal>
  );
}

const STATUS_STYLE = {
  coding: { bg: C.redBg, fg: C.red, icon: 'alert-triangle' },
  codingDay: { bg: C.amberBg, fg: C.amber, icon: 'clock' },
  free: { bg: C.greenBg, fg: C.green, icon: 'check-circle' },
  holiday: { bg: C.greenBg, fg: C.green, icon: 'check-circle' },
  invalid: { bg: C.greyBg, fg: C.body, icon: 'help-circle' },
};

function CarCard({ car, status, onCheck, onRename, onDelete }) {
  const st = STATUS_STYLE[status.kind];
  return (
    <Card style={{ padding: 0, overflow: 'hidden' }}>
      <View style={{ flexDirection: 'row', alignItems: 'center', padding: 16 }}>
        <View style={[styles.carIcon, { backgroundColor: st.bg }]}>
          <MaterialCommunityIcons name="car-side" size={26} color={st.fg} />
        </View>
        <View style={{ flex: 1, marginLeft: 14 }}>
          <T w="b" style={{ fontSize: 17 }} numberOfLines={1}>
            {car.name}
          </T>
          <T w="m" style={{ fontSize: 14, color: C.muted, marginTop: 2, letterSpacing: 0.3 }}>
            {formatPlate(car.plate)}
          </T>
        </View>
        <View style={[styles.statusPill, { backgroundColor: st.bg }]}>
          <Feather name={st.icon} size={13} color={st.fg} />
          <T w="s" style={{ fontSize: 12, color: st.fg, marginLeft: 5 }}>
            {status.label}
          </T>
        </View>
      </View>
      <View style={styles.carFoot}>
        <T style={{ flex: 1, fontSize: 13, color: status.kind === 'coding' ? C.red : C.body }}>{status.sub}</T>
        <Pressable onPress={onRename} hitSlop={8} style={styles.carAction} accessibilityLabel={`Rename ${car.name}`}>
          <Feather name="edit-2" size={17} color={C.muted} />
        </Pressable>
        <Pressable onPress={onDelete} hitSlop={8} style={styles.carAction} accessibilityLabel={`Delete ${car.name}`}>
          <Feather name="trash-2" size={17} color={C.muted} />
        </Pressable>
        <Pressable onPress={onCheck} hitSlop={6} style={styles.carCheck}>
          <T w="s" style={{ fontSize: 13, color: C.teal }}>
            Details
          </T>
          <Feather name="chevron-right" size={16} color={C.teal} />
        </Pressable>
      </View>
    </Card>
  );
}

function MyCarsScreen({ cars, holiday, onAdd, onRename, onDelete, onCheck }) {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 30000);
    return () => clearInterval(id);
  }, []);

  const rows = cars.map((car) => ({ car, status: carStatus(car.plate, now, holiday) }));
  const inCoding = rows.filter((r) => r.status.kind === 'coding').length;
  const dayLabel = `${DAY_NAME[JS_DAY_TO_SYMBOL[now.getDay()]]}, ${label12(now.getHours(), now.getMinutes())}`;

  return (
    <ScrollView contentContainerStyle={styles.page}>
      <Brand />
      <T w="b" style={[styles.h1, { marginTop: 26 }]}>
        My cars.
      </T>
      <T style={styles.sub}>Live coding status for every car you’ve saved.</T>

      {cars.length > 0 && (
        <View style={[styles.summary, inCoding ? { backgroundColor: C.redBg } : { backgroundColor: C.greenBg }]}>
          <Feather name={inCoding ? 'alert-triangle' : 'check-circle'} size={18} color={inCoding ? C.red : C.green} />
          <View style={{ flex: 1, marginLeft: 10 }}>
            <T w="s" style={{ fontSize: 15, color: inCoding ? C.red : C.green }}>
              {inCoding
                ? cars.length === 1
                  ? 'Your car is in coding now'
                  : `${inCoding} of ${cars.length} cars ${inCoding === 1 ? 'is' : 'are'} in coding now`
                : cars.length === 1
                ? 'Your car is free to drive now'
                : 'All your cars are free to drive now'}
            </T>
            <T style={{ fontSize: 12, color: C.body, marginTop: 2 }}>
              As of {dayLabel}
              {holiday ? ' · Holiday mode on' : ''}
            </T>
          </View>
        </View>
      )}

      {cars.length === 0 ? (
        <Card style={{ alignItems: 'center', paddingVertical: 30, marginTop: 20 }}>
          <View style={[styles.carIcon, { backgroundColor: C.tealSoft, width: 60, height: 60, borderRadius: 18 }]}>
            <MaterialCommunityIcons name="car-multiple" size={30} color={C.teal} />
          </View>
          <T w="b" style={{ fontSize: 18, marginTop: 14 }}>
            No cars yet
          </T>
          <T style={{ fontSize: 13, color: C.muted, marginTop: 6, textAlign: 'center', paddingHorizontal: 20 }}>
            Add your car here, or tap “Save this plate” on a coding result.
          </T>
        </Card>
      ) : (
        <View style={{ gap: 12, marginTop: 16 }}>
          {rows.map(({ car, status }) => (
            <CarCard
              key={car.id}
              car={car}
              status={status}
              onCheck={() => onCheck(car)}
              onRename={() => onRename(car)}
              onDelete={() => onDelete(car)}
            />
          ))}
        </View>
      )}

      <Pressable onPress={onAdd} style={styles.addCar}>
        <Feather name="plus" size={20} color={C.teal} />
        <T w="s" style={{ fontSize: 15, color: C.teal, marginLeft: 8 }}>
          Add a car
        </T>
      </Pressable>
      <T style={[styles.footnote, { marginTop: 14 }]}>
        Status updates every 30 seconds using the MMDA baseline (weekdays 7–10 AM and 5–8 PM). Cars are saved on this
        phone only.
      </T>
    </ScrollView>
  );
}

/** Sun/moon button that flips between light and dark mode. */
function ThemeToggle() {
  const { scheme, setMode } = useTheme();
  const dark = scheme === 'dark';
  return (
    <Pressable
      onPress={() => setMode(dark ? 'light' : 'dark')}
      hitSlop={10}
      style={styles.iconBtn}
      accessibilityLabel={dark ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      <Feather name={dark ? 'sun' : 'moon'} size={20} color={C.ink} />
    </Pressable>
  );
}

function HomeScreen({ plateText, setPlateText, city, onPickCity, onScan, onCheck, onHelp, holiday, cars }) {
  return (
    <ScrollView contentContainerStyle={styles.page} keyboardShouldPersistTaps="handled">
      <View style={styles.topRow}>
        <Brand />
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
          <ThemeToggle />
          <Pressable onPress={onHelp} hitSlop={10} style={styles.iconBtn} accessibilityLabel="How coding works">
            <Feather name="help-circle" size={20} color={C.ink} />
          </Pressable>
        </View>
      </View>

      <T w="b" style={[styles.h1, { marginTop: 26 }]}>
        Your plate.{'\n'}Your travel plan.
      </T>
      <T style={styles.sub}>Check number-coding restrictions before you drive.</T>

      <Pressable onPress={onScan} style={({ pressed }) => [styles.scanCard, pressed && { opacity: 0.92 }]}>
        <View style={styles.scanIconTile}>
          <MaterialCommunityIcons name="scan-helper" size={26} color={C.scan} />
        </View>
        <View style={{ flex: 1, marginLeft: 16 }}>
          <T w="s" style={{ fontSize: 18, color: '#fff' }}>
            Scan your plate
          </T>
          <T style={{ fontSize: 13, color: '#C3CEDA', marginTop: 4 }}>Use your camera to read the number.</T>
          <T w="b" style={{ fontSize: 10, color: C.scan, marginTop: 10, letterSpacing: 1 }}>
            CAMERA + OCR
          </T>
        </View>
        <Feather name="chevron-right" size={24} color="#fff" />
      </Pressable>

      <View style={styles.divider}>
        <View style={styles.dividerLine} />
        <T style={styles.dividerText}>or enter manually</T>
        <View style={styles.dividerLine} />
      </View>

      <T w="s" style={styles.fieldLabel}>
        Plate number
      </T>
      <View style={styles.plateInputWrap}>
        <TextInput
          value={plateText}
          onChangeText={(t) => setPlateText(t.toUpperCase())}
          placeholder="NCA 1231"
          placeholderTextColor={C.placeholder}
          autoCapitalize="characters"
          autoCorrect={false}
          maxLength={16}
          returnKeyType="done"
          onSubmitEditing={onCheck}
          style={styles.plateInput}
        />
        <Feather name="edit-2" size={20} color={C.muted} />
      </View>
      <T style={styles.helper}>Letters and numbers only. No spaces needed.</T>
      <SavedPlates cars={cars} current={plateText} onPick={setPlateText} />

      <Pressable onPress={onPickCity} style={[styles.card, styles.locationCard]}>
        <Feather name="map-pin" size={22} color={C.teal} />
        <View style={{ flex: 1, marginLeft: 14 }}>
          <T w="s" style={styles.tinyLabel}>
            CHECK LOCATION
          </T>
          <T w="m" style={{ fontSize: 15, marginTop: 2 }}>
            {city}, Metro Manila
          </T>
        </View>
        <Feather name="chevron-down" size={20} color={C.muted} />
      </Pressable>
      <T style={styles.helper}>Vehicle type: Private car{holiday ? ' · Holiday mode on' : ''}</T>

      <PrimaryButton label="Check restrictions" onPress={onCheck} disabled={!plateText.trim()} style={{ marginTop: 22 }} />
      <T w="m" style={[styles.footnote, { marginTop: 16 }]}>
        Philippines · Metro Manila example
      </T>
      <T style={styles.footnote}>Local rules and holiday suspensions may differ.</T>
    </ScrollView>
  );
}

function CityPicker({ visible, city, onSelect, onClose }) {
  const insets = useSafeAreaInsets();
  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <Pressable style={styles.sheetBackdrop} onPress={onClose} />
      <View style={[styles.sheet, { paddingBottom: insets.bottom + 12 }]}>
        <View style={styles.sheetHandle} />
        <T w="b" style={{ fontSize: 18, marginBottom: 4 }}>
          Check location
        </T>
        <T style={{ fontSize: 13, color: C.muted, marginBottom: 10 }}>All cities use the MMDA baseline schedule here.</T>
        <ScrollView style={{ maxHeight: 380 }}>
          {NCR_CITIES.map((c) => (
            <Pressable key={c} onPress={() => onSelect(c)} style={styles.sheetRow}>
              <T w={c === city ? 's' : 'r'} style={{ fontSize: 16, color: c === city ? C.teal : C.ink }}>
                {c}
              </T>
              {c === city && <Feather name="check" size={20} color={C.teal} />}
            </Pressable>
          ))}
        </ScrollView>
      </View>
    </Modal>
  );
}

// ============================================================================
//  SCAN PLATE
// ============================================================================
function ScanScreen({ onBack, onConfirm }) {
  const [permission, requestPermission] = useCameraPermissions();
  const camRef = useRef(null);
  const [box, setBox] = useState(null);
  const [phase, setPhase] = useState('aim'); // aim | reading | detected | notfound
  const [photo, setPhoto] = useState(null);
  const [torch, setTorch] = useState(false);
  const [plate, setPlate] = useState('');
  const [ocrRaw, setOcrRaw] = useState('');
  const [reads, setReads] = useState([]);
  const [agreement, setAgreement] = useState(null);
  const [step, setStep] = useState('');

  // Leave a crash note while the camera screen is open; clear it when leaving normally.
  useEffect(() => {
    markStep('opening the camera');
    return () => {
      clearStep();
    };
  }, []);

  const frame = useMemo(() => {
    if (!box) return null;
    const w = box.w * 0.8;
    const h = w * 0.42;
    return { x: (box.w - w) / 2, y: box.h * 0.43 - h / 2, w, h };
  }, [box]);

  const line = useScanLine(phase === 'aim' || phase === 'capturing' || phase === 'reading', frame ? frame.h * 0.8 : 40);

  const BURST = 3; // shots per scan; each is read separately and the results are voted

  // One shot → { plate (best guess or null), raw (for the pipeline), text (what to show) }
  const readShot = async (pic, crop) => {
    await markStep('reading a photo with ML Kit');
    const r = await recognizeWithMlKit(pic.uri, { width: pic.width, height: pic.height, crop });
    let raw = (r.text || '').trim();
    // Nothing plate-shaped inside the frame? Read the whole photo before giving up.
    if (crop && !plateCandidates(raw).length) {
      const whole = (await r.fullText()).trim();
      if (plateCandidates(whole).length) raw = whole;
    }
    return { plate: plateCandidates(raw)[0]?.plate || null, raw, text: raw.replace(/\n/g, ' / ') || '(nothing)' };
  };

  // pics = [{ uri, base64? }, ...], crop = scan frame in preview coordinates (camera only)
  const runOcr = async (pics, crop) => {
    setPhase('reading');
    try {
      const shots = [];
      for (let i = 0; i < pics.length; i += 1) {
        setStep(`Reading ${i + 1} of ${pics.length}`);
        shots.push(await readShot(pics[i], crop));
      }
      const vote = votePlate(shots.map((r) => r.plate));
      setReads(shots.map((r) => r.text));
      if (vote) {
        // keep the raw text of a shot that matches the vote, so the result keeps its OCR cleanup steps
        const match = shots.find((r) => r.plate === vote.plate);
        setOcrRaw(match ? match.raw : '');
        setAgreement(vote);
        setPlate(formatPlate(vote.plate));
        setPhase('detected');
      } else {
        const std = standardizeInput(shots[0]?.raw || '', 'ocr');
        setOcrRaw(shots[0]?.raw || '');
        setAgreement(null);
        setPlate(formatPlate(std.plate));
        setPhase('notfound');
      }
    } catch (e) {
      setPhase('notfound');
      Alert.alert('OCR failed', String(e?.message || e));
    } finally {
      markStep('scan screen open');
    }
  };

  const ensureEngine = () => {
    if (ML_KIT_AVAILABLE) return true;
    Alert.alert(
      'Scanning needs the PlateWise app',
      'Plate scanning uses Google ML Kit, which only runs in the installed PlateWise app (APK), not in Expo Go. You can still type the plate below.'
    );
    return false;
  };

  const capture = async () => {
    if (!ensureEngine()) return;
    try {
      const pics = [];
      for (let i = 0; i < BURST; i += 1) {
        setStep(`Hold steady… ${i + 1}/${BURST}`);
        setPhase('capturing');
        await markStep(`taking photo ${i + 1} of ${BURST}`);
        const pic = await camRef.current?.takePictureAsync({ quality: 0.92, shutterSound: false });
        if (!pic?.uri) throw new Error('No photo was captured.');
        pics.push(pic);
      }
      setTorch(false);
      setPhoto(pics[0].uri);
      await runOcr(pics, { boxW: box.w, boxH: box.h, ...frame, pad: 0.15 });
    } catch (e) {
      Alert.alert('Camera error', String(e?.message || e));
      setPhase('aim');
    }
  };

  const fromGallery = async () => {
    if (!ensureEngine()) return;
    await markStep('opening the photo picker');
    const picked = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 0.9,
      allowsEditing: true,
    });
    if (picked.canceled || !picked.assets?.length) return;
    setPhoto(picked.assets[0].uri);
    await runOcr([picked.assets[0]], null);
  };

  const reset = () => {
    setPhoto(null);
    setPlate('');
    setOcrRaw('');
    setReads([]);
    setAgreement(null);
    setPhase('aim');
  };

  const granted = permission?.granted;

  return (
    <ScrollView contentContainerStyle={styles.page} keyboardShouldPersistTaps="handled">
      <ScreenHeader title="Scan plate" onBack={onBack} />
      <T w="b" style={[styles.h1, { fontSize: 28, marginTop: 22 }]}>
        Align. Scan. Confirm.
      </T>
      <T style={styles.sub}>Keep the entire plate inside the frame.</T>

      <View style={styles.cameraCard} onLayout={(e) => setBox({ w: e.nativeEvent.layout.width, h: e.nativeEvent.layout.height })}>
        {granted && !photo && (
          <CameraView
            ref={camRef}
            style={StyleSheet.absoluteFill}
            facing="back"
            enableTorch={torch}
            animateShutter={false}
            onCameraReady={() => markStep('camera preview running')}
          />
        )}
        {photo && <Image source={{ uri: photo }} style={StyleSheet.absoluteFill} resizeMode="cover" />}
        {(granted || photo) && <View style={[StyleSheet.absoluteFill, { backgroundColor: 'rgba(15,34,56,0.28)' }]} />}

        <View style={styles.camTop}>
          <View style={styles.camPill}>
            <T w="b" style={styles.camPillText}>
              {photo ? 'CAPTURED PHOTO' : ML_KIT_AVAILABLE ? 'LIVE CAMERA' : 'SCANNING UNAVAILABLE'}
            </T>
          </View>
          {granted && !photo && (
            <Pressable onPress={() => setTorch((t) => !t)} hitSlop={10}>
              <Feather name={torch ? 'zap' : 'zap-off'} size={22} color="#fff" />
            </Pressable>
          )}
        </View>

        {!granted && !photo ? (
          <View style={styles.camPermission}>
            <Feather name="camera" size={30} color={C.scan} />
            <T w="s" style={{ color: '#fff', fontSize: 16, marginTop: 10 }}>
              Camera access needed
            </T>
            <T style={{ color: '#C3CEDA', fontSize: 13, marginTop: 4, textAlign: 'center' }}>
              PlateWise reads your plate from a photo. Nothing is uploaded.
            </T>
            <Pressable onPress={requestPermission} style={styles.camAllow}>
              <T w="s" style={{ color: C.navy }}>
                Allow camera
              </T>
            </Pressable>
          </View>
        ) : (
          frame && (
            <>
              <View style={[styles.frameBox, { left: frame.x, top: frame.y, width: frame.w, height: frame.h }]}>
                <ScanCorners />
                {(phase === 'aim' || phase === 'capturing' || phase === 'reading') && (
                  <Animated.View style={[styles.scanLine, { transform: [{ translateY: line }] }]} />
                )}
              </View>
              <View style={[styles.camStatus, { top: frame.y + frame.h + 14 }]}>
                {phase === 'aim' && (
                  <T style={styles.camStatusSub}>Fill the frame with the plate, hold steady, then tap the shutter.</T>
                )}
                {(phase === 'capturing' || phase === 'reading') && (
                  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                    <ActivityIndicator color={C.scan} size="small" />
                    <T w="s" style={styles.camStatusTitle}>
                      {step || 'Reading plate…'}
                    </T>
                  </View>
                )}
                {phase === 'detected' && (
                  <>
                    <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                      <Feather name="check" size={18} color={C.scan} />
                      <T w="s" style={styles.camStatusTitle}>
                        Plate detected
                      </T>
                    </View>
                    <T style={styles.camStatusSub}>
                      {agreement && agreement.total > 1
                        ? `${agreement.agree} of ${agreement.total} reads agree. Check the characters below.`
                        : 'Check the characters below to continue.'}
                    </T>
                  </>
                )}
                {phase === 'notfound' && (
                  <>
                    <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                      <Feather name="alert-circle" size={18} color="#F4B9A8" />
                      <T w="s" style={styles.camStatusTitle}>
                        No valid plate found
                      </T>
                    </View>
                    <T style={styles.camStatusSub}>Move closer, avoid glare, or type it below.</T>
                  </>
                )}
              </View>
              {phase === 'aim' && granted && (
                <Pressable onPress={capture} style={styles.shutter} accessibilityLabel="Capture plate">
                  <View style={styles.shutterInner} />
                </Pressable>
              )}
            </>
          )
        )}
      </View>

      {!ML_KIT_AVAILABLE && (
        <T style={{ fontSize: 12, color: C.red, marginTop: 10 }}>
          Scanning only works in the installed PlateWise app (it uses Google ML Kit). Type the plate below instead.
        </T>
      )}

      <T w="b" style={{ fontSize: 20, marginTop: 26 }}>
        Confirm detected plate
      </T>
      <T style={[styles.helper, { marginTop: 4 }]}>Tap the plate number to correct any mistakes.</T>
      <View style={[styles.plateInputWrap, { borderColor: C.line, marginTop: 12 }]}>
        <TextInput
          value={plate}
          onChangeText={(t) => setPlate(t.toUpperCase())}
          placeholder={phase === 'reading' || phase === 'capturing' ? 'Reading…' : 'Waiting for scan'}
          placeholderTextColor={C.placeholder}
          autoCapitalize="characters"
          autoCorrect={false}
          maxLength={16}
          style={styles.plateInput}
        />
        <Feather name="edit-2" size={20} color={C.muted} />
      </View>
      <View style={[styles.topRow, { marginTop: 12 }]}>
        <Pressable onPress={reset} hitSlop={8}>
          <T w="s" style={styles.link}>
            Scan again
          </T>
        </Pressable>
        <Pressable onPress={fromGallery} hitSlop={8}>
          <T w="s" style={styles.link}>
            Use a photo instead
          </T>
        </Pressable>
      </View>
      {reads.length > 0 && (
        <T style={[styles.footnote, { marginTop: 8, fontFamily: MONO }]}>
          {reads.map((r, i) => `Shot ${i + 1}: ${r}`).join('\n')}
        </T>
      )}

      <PrimaryButton
        label="Confirm & check"
        arrow={false}
        disabled={!plate.trim() || phase === 'reading' || phase === 'capturing'}
        onPress={() => onConfirm(plate, ocrRaw)}
        style={{ marginTop: 22 }}
      />
      <T style={[styles.footnote, { marginTop: 14 }]}>OCR can make mistakes. Always confirm your plate.</T>
    </ScrollView>
  );
}

// ============================================================================
//  CODING RESULT
// ============================================================================
function statusFor(check) {
  const { result, day, holiday, timeLabel, mode, lastDigit } = check;
  const when = mode === 'live' ? 'today' : `on ${DAY_NAME[day]}`;
  const codingDayName = result.codingDay ? DAY_NAME[result.codingDay] : null;

  if (result.verdict === VERDICT.REJECTED) {
    return {
      kind: 'invalid',
      bg: C.greyBg,
      fg: C.body,
      icon: 'help-circle',
      title: 'Plate not recognized',
      sub: 'Expected 3 letters and 4 digits, like NCA 1231.',
    };
  }
  if (result.verdict === VERDICT.CODING) {
    return {
      kind: 'restricted',
      bg: C.redBg,
      fg: C.red,
      icon: 'alert-triangle',
      title: `Restricted at ${timeLabel}`,
      sub: `Plate ending ${lastDigit} · ${codingDayName} coding`,
      hours: true,
    };
  }
  if (result.path.includes('SR')) {
    return {
      kind: 'caution',
      bg: C.amberBg,
      fg: C.amber,
      icon: 'clock',
      title: `Allowed at ${timeLabel}`,
      sub: `${codingDayName} is your coding day. Avoid these hours:`,
      hours: true,
    };
  }
  if (holiday) {
    return {
      kind: 'allowed',
      bg: C.greenBg,
      fg: C.green,
      icon: 'check-circle',
      title: 'Allowed · holiday',
      sub: `Coding is lifted on official non-working holidays. Your usual coding day is ${codingDayName}.`,
    };
  }
  const weekend = day === 'Sat' || day === 'Sun';
  return {
    kind: 'allowed',
    bg: C.greenBg,
    fg: C.green,
    icon: 'check-circle',
    title: `Allowed ${when}`,
    sub: weekend
      ? `No coding on weekends. Your coding day is ${codingDayName}.`
      : `Plate ending ${lastDigit} · your coding day is ${codingDayName}.`,
  };
}

/** Plain-language reason behind a verdict, shown under "Know why?". */
function whyText(check, st) {
  const { result, day, holiday, timeLabel, lastDigit } = check;
  const cd = result.codingDay;
  const rule = `MMDA number coding keeps private cars off major roads one weekday a week, based on the plate’s last digit. Plates ending in ${DAY_ENDINGS[cd].replace(' · ', ' or ')} are coded on ${DAY_NAME[cd]}s, 7–10 AM and 5–8 PM.`;
  const dayName = DAY_NAME[day];
  let mine;
  if (st.kind === 'restricted') mine = `Your plate ends in ${lastDigit}, and ${timeLabel} on ${dayName} falls inside the restricted hours.`;
  else if (st.kind === 'caution')
    mine = `${dayName} is your coding day, but ${timeLabel} is outside the restricted hours, so you can drive now.`;
  else if (holiday) mine = `This check is marked as a holiday, and coding is lifted on official non-working holidays.`;
  else if (day === 'Sat' || day === 'Sun') mine = `Coding doesn’t apply on weekends, so you can drive any time.`;
  else mine = `Your plate ends in ${lastDigit}, so ${dayName} isn’t your coding day. You can drive any time.`;
  return `${rule}\n\n${mine}`;
}

function ResultScreen({ check, onBack, onAnother, cars, onSave }) {
  const savedCar = cars.find((c) => c.plate === check.std.plate);
  const st = statusFor(check);
  const [why, setWhy] = useState(false);
  const toggleWhy = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setWhy((w) => !w);
  };
  const dayText = `${DAY_NAME[check.day].toUpperCase()}${check.holiday ? ' (HOLIDAY)' : ''}, ${check.timeLabel}`;
  return (
    <ScrollView contentContainerStyle={styles.page}>
      <ScreenHeader title="Coding result" onBack={onBack} />
      <T w="s" style={[styles.tinyLabel, { marginTop: 22, fontSize: 11 }]}>
        {check.mode === 'live' ? 'CHECKED NOW' : 'SCHEDULED CHECK'} · {dayText}
      </T>

      <Card style={[styles.rowBetween, { marginTop: 12 }]}>
        <View style={{ flex: 1 }}>
          <T w="b" style={{ fontSize: 26, letterSpacing: 0.3 }} numberOfLines={1}>
            {check.plateDisplay}
          </T>
          {check.valid && (
            <Pressable onPress={() => onSave(check.std.plate, savedCar)} style={styles.saveBtn} hitSlop={6}>
              <MaterialCommunityIcons name={savedCar ? 'bookmark-check' : 'bookmark-outline'} size={18} color={C.teal} />
              <T w="s" style={{ fontSize: 13, color: C.teal, marginLeft: 6 }} numberOfLines={1}>
                {savedCar ? `Saved as “${savedCar.name}” · Rename` : 'Save this plate'}
              </T>
            </Pressable>
          )}
        </View>
        <View style={{ alignItems: 'flex-end' }}>
          <T w="s" style={styles.tinyLabel}>
            LAST DIGIT
          </T>
          <T w="b" style={{ fontSize: 26, color: C.teal, marginTop: 2 }}>
            {check.lastDigit}
          </T>
        </View>
      </Card>

      <View style={[styles.statusCard, { backgroundColor: st.bg }]}>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          <Feather name={st.icon} size={24} color={st.fg} />
          <T w="b" style={{ fontSize: 22, color: st.fg, marginLeft: 10, flex: 1 }}>
            {st.title}
          </T>
        </View>
        <T w="m" style={{ fontSize: 14, color: st.fg, marginTop: 8 }}>
          {st.sub}
        </T>
        {st.kind === 'invalid' && (
          <T style={{ fontSize: 13, color: C.body, marginTop: 8 }}>
            {check.std.plate ? `We read “${check.std.plate}”. Check the plate and try again.` : 'No plate was entered.'}
          </T>
        )}
        {st.hours && (
          <>
            <View style={[styles.statusRule, { backgroundColor: st.fg }]} />
            <T w="b" style={{ fontSize: 11, letterSpacing: 0.8, color: st.fg }}>
              RESTRICTED HOURS
            </T>
            <RestrictedHours color={st.fg} />
          </>
        )}
      </View>

      {st.kind !== 'invalid' && (
        <View style={[styles.card, styles.whyCard]}>
          <Pressable
            onPress={toggleWhy}
            style={styles.whyHead}
            accessibilityRole="button"
            accessibilityState={{ expanded: why }}
          >
            <Feather name="help-circle" size={20} color={C.teal} />
            <T w="s" style={{ flex: 1, marginLeft: 10, fontSize: 15, color: C.teal }}>
              Know why?
            </T>
            <Feather name={why ? 'chevron-up' : 'chevron-down'} size={20} color={C.teal} />
          </Pressable>

          {why && (
            <View style={styles.whyBody}>
              <T style={styles.guideLine}>{whyText(check, st)}</T>

              <T w="s" style={[styles.tinyLabel, { marginTop: 18, marginBottom: 8 }]}>
                PLATE ENDINGS BY DAY
              </T>
              <WeekGuide highlightDay={check.result.codingDay} restricted={st.kind === 'restricted'} />

              <View style={{ flexDirection: 'row', alignItems: 'center', marginTop: 16 }}>
                <Feather name="map-pin" size={16} color={C.muted} />
                <T style={{ fontSize: 13, color: C.muted, marginLeft: 8, flex: 1 }}>
                  {check.city}, Metro Manila · MMDA baseline · Private car
                </T>
              </View>

              <InfoNote>Holidays, exemptions and local rules may differ. Verify official advisories before travelling.</InfoNote>
              <T style={[styles.footnote, { marginTop: 10 }]}>
                Baseline: MMDA UVVRP, weekdays 7–10 AM and 5–8 PM (Mitra, 2026).
              </T>
            </View>
          )}
        </View>
      )}

      <PrimaryButton label="Check another plate" arrow={false} onPress={onAnother} style={{ marginTop: 20 }} />
    </ScrollView>
  );
}

// ============================================================================
//  SCHEDULE TAB
// ============================================================================
function ScheduleScreen({ plateText, setPlateText, holiday, setHoliday, onRunScheduled, cars }) {
  const std = standardizeInput(plateText, 'manual');
  const group = plateGroup(std.chars);
  const codingDay = group ? GROUP_CODING_DAY[group] : null;
  const today = JS_DAY_TO_SYMBOL[new Date().getDay()];

  const [day, setDay] = useState(today === 'Sat' || today === 'Sun' ? 'Mon' : today);
  const [hour12, setHour12] = useState(8);
  const [minute, setMinute] = useState(0);
  const [mer, setMer] = useState('AM');
  const presets = [
    ['7:00 AM', 7, 0, 'AM'],
    ['10:00 AM', 10, 0, 'AM'],
    ['10:01 AM', 10, 1, 'AM'],
    ['12:00 PM', 12, 0, 'PM'],
    ['5:00 PM', 5, 0, 'PM'],
    ['8:01 PM', 8, 1, 'PM'],
  ];

  return (
    <ScrollView contentContainerStyle={styles.page} keyboardShouldPersistTaps="handled">
      <Brand />
      <T w="b" style={[styles.h1, { marginTop: 26 }]}>
        Your coding week.
      </T>
      <T style={styles.sub}>
        {group ? `Plate ${formatPlate(std.plate)} · coding every ${DAY_NAME[codingDay]}.` : 'Enter a plate to see its coding days.'}
      </T>

      <View style={[styles.plateInputWrap, { marginTop: 18 }]}>
        <TextInput
          value={plateText}
          onChangeText={(t) => setPlateText(t.toUpperCase())}
          placeholder="NCA 1231"
          placeholderTextColor={C.placeholder}
          autoCapitalize="characters"
          autoCorrect={false}
          maxLength={16}
          style={[styles.plateInput, { fontSize: 20 }]}
        />
        <Feather name="edit-2" size={18} color={C.muted} />
      </View>
      {!!plateText.trim() && !group && <T style={[styles.helper, { color: C.red }]}>Not a valid plate (3 letters + 4 digits).</T>}
      <SavedPlates cars={cars} current={plateText} onPick={setPlateText} />

      <Card style={{ marginTop: 18, paddingVertical: 4 }}>
        {WEEK.map((d, i) => {
          const isCoding = d === codingDay;
          const isToday = d === today;
          return (
            <View key={d} style={[styles.weekRow, i < WEEK.length - 1 && { borderBottomWidth: 1, borderBottomColor: C.line }]}>
              <View style={{ width: 112, flexDirection: 'row', alignItems: 'center' }}>
                <T w="s" style={{ fontSize: 15 }}>
                  {DAY_NAME[d]}
                </T>
                {isToday && (
                  <View style={styles.todayPill}>
                    <T w="b" style={{ fontSize: 9, color: C.teal }}>
                      TODAY
                    </T>
                  </View>
                )}
              </View>
              <View style={{ flex: 1, alignItems: 'flex-end' }}>
                {isCoding ? (
                  <>
                    <T w="s" style={{ fontSize: 14, color: holiday && isToday ? C.green : C.red }}>
                      {holiday && isToday ? 'Lifted (holiday)' : 'Coding'}
                    </T>
                    <T style={{ fontSize: 12, color: C.muted }}>7–10 AM · 5–8 PM</T>
                  </>
                ) : (
                  <T style={{ fontSize: 14, color: C.muted }}>{group ? 'No coding' : '—'}</T>
                )}
              </View>
            </View>
          );
        })}
      </Card>

      <Card style={[styles.rowBetween, { marginTop: 14 }]}>
        <View style={{ flex: 1, paddingRight: 12 }}>
          <T w="s" style={{ fontSize: 15 }}>
            Today is a holiday
          </T>
          <T style={{ fontSize: 12, color: C.muted, marginTop: 2 }}>
            Coding is lifted on official non-working holidays. Turn this on to treat today as one.
          </T>
        </View>
        <Switch value={holiday} onValueChange={setHoliday} trackColor={{ true: C.teal, false: C.switchOff }} thumbColor="#fff" />
      </Card>

      <T w="b" style={{ fontSize: 18, marginTop: 26 }}>
        Check a specific time
      </T>
      <T style={[styles.helper, { marginTop: 4 }]}>Pick any day and time to run the same check.</T>
      <Card style={{ marginTop: 12 }}>
        <T w="s" style={styles.tinyLabel}>
          DAY
        </T>
        <View style={[styles.wrap, { marginTop: 8 }]}>
          {DAY_SYMBOLS.map((d) => (
            <Chip key={d} label={d} active={day === d} onPress={() => setDay(d)} />
          ))}
        </View>
        <T w="s" style={[styles.tinyLabel, { marginTop: 16 }]}>
          TIME
        </T>
        <View style={[styles.wrap, { marginTop: 8, alignItems: 'center' }]}>
          <Stepper value={String(hour12)} onMinus={() => setHour12((h) => ((h + 10) % 12) + 1)} onPlus={() => setHour12((h) => (h % 12) + 1)} />
          <T w="b" style={{ fontSize: 20 }}>
            :
          </T>
          <Stepper value={pad(minute)} onMinus={() => setMinute((m) => (m + 59) % 60)} onPlus={() => setMinute((m) => (m + 1) % 60)} />
          <Chip label="AM" active={mer === 'AM'} onPress={() => setMer('AM')} />
          <Chip label="PM" active={mer === 'PM'} onPress={() => setMer('PM')} />
        </View>
        <View style={[styles.wrap, { marginTop: 10 }]}>
          {presets.map(([label, h, m, me]) => (
            <Chip
              key={label}
              label={label}
              active={hour12 === h && minute === m && mer === me}
              onPress={() => {
                setHour12(h);
                setMinute(m);
                setMer(me);
              }}
            />
          ))}
        </View>
        <T style={[styles.helper, { marginTop: 10 }]}>
          Time symbol: <T w="b">{timeToSymbol(to24(hour12, mer), minute)}</T>{' '}
          {timeToSymbol(to24(hour12, mer), minute) === 'p' ? '(peak hours)' : '(non-peak)'}
        </T>
      </Card>
      <PrimaryButton
        label="Check this time"
        disabled={!plateText.trim()}
        onPress={() => onRunScheduled({ day: day === 'Hol' ? today : day, holiday: day === 'Hol', h24: to24(hour12, mer), m: minute })}
        style={{ marginTop: 16 }}
      />
    </ScrollView>
  );
}

// ============================================================================
//  GUIDE TAB
// ============================================================================
function AppearanceCard() {
  const { mode, setMode } = useTheme();
  const options = [
    ['system', 'System'],
    ['light', 'Light'],
    ['dark', 'Dark'],
  ];
  return (
    <Card style={{ marginTop: 22 }}>
      <T w="s" style={{ fontSize: 15 }}>
        Appearance
      </T>
      <View style={[styles.wrap, { marginTop: 12 }]}>
        {options.map(([key, label]) => (
          <Chip key={key} label={label} active={mode === key} onPress={() => setMode(key)} />
        ))}
      </View>
    </Card>
  );
}

function GuideScreen() {
  return (
    <ScrollView contentContainerStyle={styles.page}>
      <View style={styles.topRow}>
        <Brand />
        <ThemeToggle />
      </View>
      <T w="b" style={[styles.h1, { marginTop: 26 }]}>
        How coding works.
      </T>
      <T style={styles.sub}>The MMDA number-coding scheme (UVVRP) limits private cars by their plate’s last digit.</T>

      <Card style={{ marginTop: 18 }}>
        <T w="s" style={styles.tinyLabel}>
          RESTRICTED HOURS · WEEKDAYS
        </T>
        <RestrictedHours color={C.ink} />
      </Card>

      <T w="b" style={{ fontSize: 18, marginTop: 22, marginBottom: 10 }}>
        Which day is yours?
      </T>
      <WeekGuide highlightDay={null} />

      <Card style={{ marginTop: 14, gap: 10 }}>
        <T w="s" style={{ fontSize: 15 }}>
          When coding doesn’t apply
        </T>
        <T style={styles.guideLine}>• Saturdays and Sundays</T>
        <T style={styles.guideLine}>• Official non-working holidays (turn on Holiday in Schedule)</T>
        <T style={styles.guideLine}>• Outside 7–10 AM and 5–8 PM on your coding day</T>
        <T style={[styles.guideLine, { color: C.muted }]}>
          Public utility vehicles and other exempt types follow separate rules and aren’t covered here.
        </T>
      </Card>

      <AppearanceCard />

      <Card style={{ marginTop: 22, backgroundColor: C.navy, borderColor: C.navy }}>
        <T w="b" style={{ fontSize: 10, color: C.scan, letterSpacing: 1 }}>
          COSC 302 · AUTOMATA AND LANGUAGE THEORY
        </T>
        <T w="s" style={{ color: '#fff', fontSize: 15, marginTop: 8 }}>
          Group 12 · BSCS 3-5
        </T>
        <T style={{ color: '#C3CEDA', fontSize: 13, marginTop: 6, lineHeight: 20 }}>
          Bulan, Zaldy Marcus B.{'\n'}Del Mundo, M. A.{'\n'}Librado, C. T.{'\n'}Paz, L. H.
        </T>
        <T style={{ color: '#93A3B5', fontSize: 12, marginTop: 10 }}>
          Polytechnic University of the Philippines · CCIS{'\n'}Prof. Ria A. Sagum
        </T>
      </Card>
      <T style={[styles.footnote, { marginTop: 12 }]}>Plate reading powered by Google ML Kit (on-device).</T>
    </ScrollView>
  );
}

// ============================================================================
//  BOTTOM TABS
// ============================================================================
function TabBar({ tab, onTab }) {
  const insets = useSafeAreaInsets();
  const tabs = [
    ['check', 'Check plate', (c) => <MaterialCommunityIcons name="car-outline" size={24} color={c} />],
    ['cars', 'My cars', (c) => <MaterialCommunityIcons name="car-multiple" size={23} color={c} />],
    ['schedule', 'Schedule', (c) => <Feather name="calendar" size={22} color={c} />],
    ['guide', 'Guide', (c) => <Feather name="help-circle" size={22} color={c} />],
  ];
  return (
    <View style={[styles.tabBarWrap, { paddingBottom: Math.max(insets.bottom, 10) }]}>
      <View style={styles.tabBar}>
        {tabs.map(([key, label, icon]) => {
          const on = tab === key;
          return (
            <Pressable
              key={key}
              onPress={() => onTab(key)}
              style={styles.tabItem}
              accessibilityRole="tab"
              accessibilityState={{ selected: on }}
            >
              <View style={[styles.tabIcon, on && { backgroundColor: C.teal }]}>{icon(on ? '#fff' : C.navMuted)}</View>
              <T w={on ? 's' : 'm'} style={{ fontSize: 11, color: on ? '#fff' : C.navMuted, marginTop: 4 }} numberOfLines={1}>
                {label}
              </T>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

// ============================================================================
//  MAIN SHELL
// ============================================================================
function MainApp() {
  const [tab, setTab] = useState('check');
  const [route, setRoute] = useState('home'); // home | scan | result
  const [plateText, setPlateText] = useState('');
  const [city, setCity] = useState('Quezon City');
  const [cityOpen, setCityOpen] = useState(false);
  const [holiday, setHoliday] = useState(false);
  const [check, setCheck] = useState(null);
  const [cars, setCars] = useState([]);
  const [editor, setEditor] = useState(null); // { mode: 'add'|'save'|'rename', id?, plate, name }

  // Load saved cars (migrating plates saved by the older version) and the last plate checked
  useEffect(() => {
    AsyncStorage.multiGet([CARS_KEY, SAVED_KEY, LAST_PLATE_KEY])
      .then(([[, carsJson], [, oldJson], [, last]]) => {
        let list = null;
        try {
          const parsed = JSON.parse(carsJson || 'null');
          if (Array.isArray(parsed)) list = parsed.filter((c) => c && typeof c.plate === 'string');
        } catch {
          /* ignore bad data */
        }
        if (!list) {
          let old = [];
          try {
            old = JSON.parse(oldJson || '[]');
          } catch {
            old = [];
          }
          list = (Array.isArray(old) ? old : [])
            .filter((p) => typeof p === 'string')
            .map((p, i) => ({ id: `m${i}-${p}`, plate: p, name: `Car ${i + 1}` }));
          AsyncStorage.setItem(CARS_KEY, JSON.stringify(list)).catch(() => {});
        }
        setCars(list);
        if (last) setPlateText((t) => t || formatPlate(last));
      })
      .catch(() => {});
  }, []);

  const persistCars = (list) => {
    setCars(list);
    AsyncStorage.setItem(CARS_KEY, JSON.stringify(list)).catch(() => {});
  };
  const submitCar = ({ id, plate, name }) => {
    if (id) persistCars(cars.map((c) => (c.id === id ? { ...c, name, plate } : c)));
    else if (cars.length >= MAX_SAVED) {
      Alert.alert('Garage full', `You can save up to ${MAX_SAVED} cars. Remove one in My cars first.`);
      return;
    } else persistCars([...cars, { id: `${Date.now()}-${Math.round(Math.random() * 1e6)}`, plate, name }]);
    setEditor(null);
  };
  const deleteCar = (car) =>
    Alert.alert('Remove car?', `Remove “${car.name}” (${formatPlate(car.plate)}) from My cars?`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Remove', style: 'destructive', onPress: () => persistCars(cars.filter((c) => c.id !== car.id)) },
    ]);
  const openSave = (plate, existing) =>
    setEditor(
      existing
        ? { mode: 'rename', id: existing.id, plate: existing.plate, name: existing.name }
        : { mode: 'save', plate, name: cars.length ? '' : 'My car' }
    );
  const rememberLast = (c) => {
    if (c.valid) AsyncStorage.setItem(LAST_PLATE_KEY, c.std.plate).catch(() => {});
  };


  const runLive = (raw, source, ocrRaw) => {
    const now = new Date();
    // If the user kept the OCR reading unchanged, run the pipeline on the raw OCR text so the
    // OCR standardization steps are applied; if they corrected it, use what they typed.
    const useOcr =
      source === 'ocr' && !!ocrRaw && standardizeInput(raw, 'manual').plate === standardizeInput(ocrRaw, 'ocr').plate;
    const c = buildCheck({
      raw: useOcr ? ocrRaw : raw,
      source: useOcr ? 'ocr' : 'manual',
      day: JS_DAY_TO_SYMBOL[now.getDay()],
      h24: now.getHours(),
      m: now.getMinutes(),
      holiday,
      mode: 'live',
      city,
    });
    rememberLast(c);
    setCheck(c);
    setTab('check');
    setRoute('result');
  };

  const runScheduled = ({ day, holiday: hol, h24, m }) => {
    const c = buildCheck({ raw: plateText, source: 'manual', day, h24, m, holiday: hol, mode: 'scheduled', city });
    rememberLast(c);
    setCheck(c);
    setTab('check');
    setRoute('result');
  };

  useEffect(() => {
    const sub = BackHandler.addEventListener('hardwareBackPress', () => {
      if (tab === 'check' && route !== 'home') {
        setRoute('home');
        return true;
      }
      if (tab !== 'check') {
        setTab('check');
        return true;
      }
      return false;
    });
    return () => sub.remove();
  }, [tab, route]);

  let body;
  if (tab === 'schedule') {
    body = (
      <ScheduleScreen
        plateText={plateText}
        setPlateText={setPlateText}
        holiday={holiday}
        setHoliday={setHoliday}
        onRunScheduled={runScheduled}
        cars={cars}
      />
    );
  } else if (tab === 'guide') {
    body = <GuideScreen />;
  } else if (tab === 'cars') {
    body = (
      <MyCarsScreen
        cars={cars}
        holiday={holiday}
        onAdd={() => setEditor({ mode: 'add', plate: '', name: cars.length ? '' : 'My car' })}
        onRename={(car) => setEditor({ mode: 'rename', id: car.id, plate: car.plate, name: car.name })}
        onDelete={deleteCar}
        onCheck={(car) => {
          setPlateText(formatPlate(car.plate));
          runLive(car.plate, 'manual');
        }}
      />
    );
  } else if (route === 'scan') {
    body = (
      <ScanScreen
        onBack={() => setRoute('home')}
        onConfirm={(plate, ocrRaw) => {
          setPlateText(plate);
          runLive(plate, 'ocr', ocrRaw);
        }}
      />
    );
  } else if (route === 'result' && check) {
    body = (
      <ResultScreen
        check={check}
        cars={cars}
        onSave={openSave}
        onBack={() => setRoute('home')}
        onAnother={() => {
          setPlateText('');
          setRoute('home');
        }}
      />
    );
  } else {
    body = (
      <HomeScreen
        plateText={plateText}
        setPlateText={setPlateText}
        city={city}
        holiday={holiday}
        onPickCity={() => setCityOpen(true)}
        onScan={() => setRoute('scan')}
        onCheck={() => plateText.trim() && runLive(plateText, 'manual')}
        onHelp={() => setTab('guide')}
        cars={cars}
      />
    );
  }

  return (
    <SafeAreaView style={styles.fill} edges={['top']}>
      <View style={styles.fill}>{body}</View>
      <TabBar
        tab={tab}
        onTab={(t) => {
          if (t === 'check' && (tab === 'check' || route === 'scan')) setRoute('home'); // re-tap returns home
          setTab(t);
        }}
      />
      <CarEditor editor={editor} cars={cars} onClose={() => setEditor(null)} onSubmit={submitCar} />
      <CityPicker
        visible={cityOpen}
        city={city}
        onClose={() => setCityOpen(false)}
        onSelect={(c) => {
          setCity(c);
          setCityOpen(false);
        }}
      />
    </SafeAreaView>
  );
}

// ============================================================================
//  ERROR SAFETY NET
// ============================================================================
/** Shows a JavaScript error on screen instead of closing the app. */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error) {
    markStep(`screen error: ${String(error?.message || error)}`);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <SafeAreaView style={[styles.fill, { padding: 24, justifyContent: 'center' }]}>
        <Feather name="alert-triangle" size={34} color={C.red} />
        <T w="b" style={{ fontSize: 22, marginTop: 12 }}>
          Something went wrong
        </T>
        <T style={[styles.sub, { marginTop: 8 }]}>Please send this message to the developers:</T>
        <Text selectable style={[styles.footnote, { fontFamily: MONO, marginTop: 10, color: C.body }]}>
          {String(this.state.error?.message || this.state.error)}
        </Text>
        <PrimaryButton
          label="Back to the app"
          arrow={false}
          style={{ marginTop: 24 }}
          onPress={() => {
            clearStep();
            this.setState({ error: null });
          }}
        />
      </SafeAreaView>
    );
  }
}

// Uncaught JavaScript errors close a release app; show them instead.
if (global.ErrorUtils?.setGlobalHandler) {
  global.ErrorUtils.setGlobalHandler((error, isFatal) => {
    const msg = String(error?.message || error);
    markStep(`app error: ${msg}`);
    Alert.alert(isFatal ? 'PlateWise hit an error' : 'Something went wrong', msg);
  });
}

// ============================================================================
//  ROOT
// ============================================================================
const ONBOARD_KEY = 'platewise.onboarded.v1';

export default function App() {
  const [fontsLoaded] = useFonts({
    Inter_400Regular: require('@expo-google-fonts/inter/400Regular/Inter_400Regular.ttf'),
    Inter_500Medium: require('@expo-google-fonts/inter/500Medium/Inter_500Medium.ttf'),
    Inter_600SemiBold: require('@expo-google-fonts/inter/600SemiBold/Inter_600SemiBold.ttf'),
    Inter_700Bold: require('@expo-google-fonts/inter/700Bold/Inter_700Bold.ttf'),
    Inter_800ExtraBold: require('@expo-google-fonts/inter/800ExtraBold/Inter_800ExtraBold.ttf'),
  });
  const [splashDone, setSplashDone] = useState(false);
  const [onboarded, setOnboarded] = useState(null); // null = still reading storage

  // Light / dark mode: follows the phone unless the user picked one.
  const systemScheme = useColorScheme();
  const [mode, setModeState] = useState('system');
  useEffect(() => {
    AsyncStorage.getItem(THEME_KEY)
      .then((v) => {
        if (v === 'light' || v === 'dark' || v === 'system') setModeState(v);
      })
      .catch(() => {});
  }, []);
  const scheme = mode === 'system' ? (systemScheme === 'dark' ? 'dark' : 'light') : mode;
  applyTheme(scheme); // swaps C and styles before this render's children read them
  const theme = useMemo(
    () => ({
      mode,
      scheme,
      setMode: (m) => {
        setModeState(m);
        AsyncStorage.setItem(THEME_KEY, m).catch(() => {});
      },
    }),
    [mode, scheme]
  );

  // If the app closed unexpectedly last time, say where.
  useEffect(() => {
    AsyncStorage.getItem(CRASH_KEY)
      .then((step) => {
        if (!step) return;
        clearStep();
        setTimeout(
          () =>
            Alert.alert(
              'PlateWise closed unexpectedly',
              `Last time, the app closed while ${step}.\n\nSend this message to the developers so they can fix it.`
            ),
          1800
        );
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    AsyncStorage.getItem(ONBOARD_KEY)
      .then((v) => setOnboarded(v === '1'))
      .catch(() => setOnboarded(false));
    const t = setTimeout(() => setSplashDone(true), 1400);
    return () => clearTimeout(t);
  }, []);

  useEffect(() => {
    if (fontsLoaded) SplashScreen.hideAsync().catch(() => {});
  }, [fontsLoaded]);

  const finishOnboarding = () => {
    setOnboarded(true);
    AsyncStorage.setItem(ONBOARD_KEY, '1').catch(() => {});
  };

  let screen;
  if (!fontsLoaded) screen = null;
  else if (!splashDone || onboarded === null) screen = <SplashView />;
  else if (!onboarded) screen = <OnboardingScreen onStart={finishOnboarding} />;
  else
    screen = (
      <ErrorBoundary>
        <MainApp />
      </ErrorBoundary>
    );

  return (
    <ThemeContext.Provider value={theme}>
      <SafeAreaProvider>
        <StatusBar style={scheme === 'dark' ? 'light' : 'dark'} />
        <View style={[styles.fill, { backgroundColor: C.bg }]}>{screen}</View>
      </SafeAreaProvider>
    </ThemeContext.Provider>
  );
}

// ============================================================================
//  STYLES
// ============================================================================
// Built from the live palette C; rebuilt by applyTheme when light/dark mode changes.
let styles = makeStyles();
function makeStyles() {
  return StyleSheet.create({
  fill: { flex: 1, backgroundColor: C.bg },
  iconBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: C.card,
    borderWidth: 1,
    borderColor: C.cardLine,
  },
  page: { paddingHorizontal: 20, paddingTop: 14, paddingBottom: 36 },
  brand: { flexDirection: 'row', alignItems: 'center' },
  topRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  rowBetween: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  wrap: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },

  h1: { fontSize: 31, lineHeight: 39, letterSpacing: -0.6, color: C.ink },
  sub: { fontSize: 14, color: C.muted, marginTop: 10, lineHeight: 20 },
  helper: { fontSize: 12, color: C.muted, marginTop: 8 },
  footnote: { fontSize: 11, color: C.muted, lineHeight: 17 },
  tinyLabel: { fontSize: 10, color: C.muted, letterSpacing: 0.8 },
  fieldLabel: { fontSize: 14, marginBottom: 10 },
  link: { fontSize: 13, color: C.teal },

  card: { backgroundColor: C.card, borderRadius: 16, padding: 16, borderWidth: 1, borderColor: C.cardLine },

  primaryBtn: {
    height: 56,
    borderRadius: 16,
    backgroundColor: C.teal,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  primaryBtnText: { color: '#fff', fontSize: 16 },
  primaryBtnArrow: { position: 'absolute', right: 20 },

  screenHeader: { flexDirection: 'row', alignItems: 'center', marginTop: 4 },

  // hero / onboarding
  heroCard: { backgroundColor: C.navy, borderRadius: 24, overflow: 'hidden', padding: 18 },
  heroCenter: { alignItems: 'center', justifyContent: 'center' },
  heroFrame: { padding: 16, alignItems: 'center', justifyContent: 'center' },
  plate: {
    backgroundColor: '#fff',
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#CBD3DC',
    alignItems: 'center',
    paddingTop: 8,
    paddingBottom: 6,
  },
  // the plate graphic is always a white plate, so its text keeps light-mode colors
  plateTop: { fontSize: 9, color: LIGHT.muted, letterSpacing: 1 },
  plateText: { color: LIGHT.ink, letterSpacing: 0.5, marginTop: 2 },
  scanLine: { position: 'absolute', left: -6, right: -6, height: 2, backgroundColor: C.scan, top: '50%' },

  // home
  scanCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.navy,
    borderRadius: 22,
    padding: 18,
    marginTop: 24,
  },
  scanIconTile: {
    width: 52,
    height: 52,
    borderRadius: 14,
    backgroundColor: 'rgba(255,255,255,0.08)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  divider: { flexDirection: 'row', alignItems: 'center', marginVertical: 22 },
  dividerLine: { flex: 1, height: 1, backgroundColor: C.line },
  dividerText: { fontSize: 12, color: C.muted, marginHorizontal: 12 },
  plateInputWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: C.tealLine,
    borderRadius: 16,
    backgroundColor: C.card,
    paddingHorizontal: 16,
  },
  plateInput: { flex: 1, minWidth: 0, fontFamily: F.b, fontSize: 26, color: C.ink, paddingVertical: 14, letterSpacing: 0.5 },
  locationCard: { flexDirection: 'row', alignItems: 'center', marginTop: 20 },

  // sheet
  sheetBackdrop: { flex: 1, backgroundColor: C.backdrop },
  sheet: { backgroundColor: C.card, borderTopLeftRadius: 24, borderTopRightRadius: 24, paddingHorizontal: 20, paddingTop: 10 },
  sheetHandle: { alignSelf: 'center', width: 40, height: 4, borderRadius: 2, backgroundColor: C.line, marginBottom: 14 },
  sheetRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: C.rowLine,
  },

  // scan
  cameraCard: { height: 310, borderRadius: 24, backgroundColor: C.navy, overflow: 'hidden', marginTop: 20 },
  camTop: {
    position: 'absolute',
    top: 14,
    left: 14,
    right: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    zIndex: 2,
  },
  camPill: { backgroundColor: 'rgba(255,255,255,0.12)', borderRadius: 999, paddingHorizontal: 12, paddingVertical: 6 },
  camPillText: { fontSize: 10, color: '#E3EAF2', letterSpacing: 0.6 },
  camPermission: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 30 },
  camAllow: { marginTop: 16, backgroundColor: '#fff', borderRadius: 12, paddingHorizontal: 18, paddingVertical: 10 },
  frameBox: { position: 'absolute' },
  camStatus: { position: 'absolute', left: 16, right: 16, alignItems: 'center' },
  camStatusTitle: { color: '#fff', fontSize: 15 },
  camStatusSub: { color: '#C3CEDA', fontSize: 12, marginTop: 4, textAlign: 'center' },
  shutter: {
    position: 'absolute',
    bottom: 14,
    alignSelf: 'center',
    width: 58,
    height: 58,
    borderRadius: 29,
    borderWidth: 3,
    borderColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  shutterInner: { width: 44, height: 44, borderRadius: 22, backgroundColor: '#fff' },

  // result
  statusCard: { borderRadius: 18, padding: 18, marginTop: 14 },
  statusRule: { height: 1, opacity: 0.18, marginVertical: 14 },
  hoursRow: { flexDirection: 'row', marginTop: 8 },
  hoursTime: { fontSize: 19 },
  hoursLabel: { fontSize: 12, marginTop: 3, opacity: 0.85 },
  weekGuide: {
    flexDirection: 'row',
    backgroundColor: C.card,
    borderRadius: 16,
    padding: 6,
    borderWidth: 1,
    borderColor: C.cardLine,
  },
  weekCell: { flex: 1, alignItems: 'center', paddingVertical: 12, borderRadius: 12 },
  weekDay: { fontSize: 11, color: C.muted, letterSpacing: 0.5 },
  weekEnd: { fontSize: 15, color: C.ink, marginTop: 6 },
  infoNote: { flexDirection: 'row', backgroundColor: C.infoBg, borderRadius: 14, padding: 14, marginTop: 14, gap: 10 },
  infoText: { flex: 1, fontSize: 13, color: C.body, lineHeight: 19 },
  whyCard: { marginTop: 14, padding: 0, overflow: 'hidden' },
  whyHead: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 15 },
  whyBody: { paddingHorizontal: 16, paddingBottom: 16, borderTopWidth: 1, borderTopColor: C.rowLine, paddingTop: 14 },


  // saved plates
  savedChip: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 999,
    borderWidth: 1,
    borderColor: C.line,
    backgroundColor: C.card,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  savedChipOn: { backgroundColor: C.teal, borderColor: C.teal },
  sheetInput: {
    borderWidth: 1.5,
    borderColor: C.tealLine,
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontFamily: F.m,
    fontSize: 17,
    color: C.ink,
    marginTop: 6,
    backgroundColor: C.card,
  },

  // my cars
  carIcon: { width: 48, height: 48, borderRadius: 14, alignItems: 'center', justifyContent: 'center' },
  statusPill: { flexDirection: 'row', alignItems: 'center', borderRadius: 999, paddingHorizontal: 10, paddingVertical: 6 },
  carFoot: {
    flexDirection: 'row',
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: C.line,
    paddingHorizontal: 16,
    paddingVertical: 10,
  },
  carAction: { paddingHorizontal: 8, paddingVertical: 4 },
  carCheck: { flexDirection: 'row', alignItems: 'center', marginLeft: 6 },
  summary: { flexDirection: 'row', alignItems: 'center', borderRadius: 16, padding: 14, marginTop: 18 },
  addCar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    height: 54,
    borderRadius: 16,
    borderWidth: 1.5,
    borderColor: C.tealLine,
    borderStyle: 'dashed',
    marginTop: 16,
  },
  saveBtn: { flexDirection: 'row', alignItems: 'center', marginTop: 10, alignSelf: 'flex-start' },

  // schedule
  weekRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 12 },
  todayPill: { backgroundColor: C.tealSoft, borderRadius: 6, paddingHorizontal: 6, paddingVertical: 2, marginLeft: 6 },
  stepper: { flexDirection: 'row', alignItems: 'center', borderWidth: 1, borderColor: C.line, borderRadius: 12, overflow: 'hidden' },
  stepBtn: { paddingHorizontal: 12, paddingVertical: 9, backgroundColor: C.surfaceAlt },
  stepValue: { minWidth: 38, textAlign: 'center', fontSize: 17 },
  chip: { paddingVertical: 8, paddingHorizontal: 13, borderRadius: 999, borderWidth: 1, borderColor: C.line, backgroundColor: C.card },
  chipActive: { backgroundColor: C.teal, borderColor: C.teal },
  chipText: { fontSize: 13, color: C.ink },

  // guide
  guideLine: { fontSize: 13, color: C.body, lineHeight: 19 },

  // tabs
  tabBarWrap: { backgroundColor: C.bg, paddingHorizontal: 14, paddingTop: 6 },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: C.nav,
    borderRadius: 22,
    paddingVertical: 9,
    paddingHorizontal: 4,
    shadowColor: '#0F2238',
    shadowOpacity: 0.18,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
    elevation: 8,
  },
  tabItem: { flex: 1, alignItems: 'center' },
  tabIcon: { width: 54, height: 32, borderRadius: 16, alignItems: 'center', justifyContent: 'center' },
  });
}
