/**
 * ============================================================================
 *  DFAEngine.js — Deterministic Finite Automaton for MMDA Number Coding Rules
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
 *    Pure automata layer. No UI, no OCR, no clock — it only receives an
 *    already-standardized array of input symbols and steps through the
 *    transition function δ one symbol at a time, exactly as in Figure 1
 *    (State diagram of M) of the proposal.
 *
 *  Formal definition  M = (Q, Σ, δ, q0, F)
 *    Q  = { S0, S1, S2, S3, S4, S5, S6, G1, G2, G3, G4, G5, SR, SN, SC, SA, SX }
 *    Σ  = L ∪ D ∪ Day ∪ Time
 *         L    = { A, B, ..., Z }                       (any letter)
 *         D    = { 0, 1, ..., 9 }                       (any digit)
 *         Day  = { Mon, Tue, Wed, Thu, Fri, Sat, Sun, Hol }  (one symbol each)
 *         Time = { p, o }   p = inside the city's coding hours, o = otherwise
 *                           (MMDA: 7:00–10:00 a.m. or 5:00–8:00 p.m.; Makati: 7:00 a.m.–7:00 p.m.)
 *         The city and holiday calendar only choose which Day/Time symbol is fed in
 *         (CodingRules.js, preprocessing). They never change Q, Σ or δ.
 *    q0 = S0
 *    F  = { SC, SA }        SC = S_Coding, SA = S_Allowed
 *    SX = trap state (every undrawn transition in Figure 1 goes here)
 *
 *  Accepted strings look like:   L L L D D D D Day Time
 *    e.g.  [A, B, C, 1, 2, 3, 4, Thu, p]
 * ============================================================================
 */

// ---------------------------------------------------------------------------
// States (Q)
// ---------------------------------------------------------------------------
export const STATES = Object.freeze({
  S0: 'S0',
  S1: 'S1',
  S2: 'S2',
  S3: 'S3',
  S4: 'S4',
  S5: 'S5',
  S6: 'S6',
  G1: 'G1',
  G2: 'G2',
  G3: 'G3',
  G4: 'G4',
  G5: 'G5',
  SR: 'SR', // coding (restricted) day for this plate
  SN: 'SN', // not a coding day for this plate
  SC: 'SC', // S_Coding  — accepting, coding restriction applies
  SA: 'SA', // S_Allowed — accepting, vehicle may travel
  SX: 'SX', // trap state
});

export const START_STATE = STATES.S0;
export const ACCEPTING_STATES = Object.freeze([STATES.SC, STATES.SA]);

// Human-readable labels used by the UI
export const STATE_LABELS = Object.freeze({
  S0: 'Start',
  S1: '1 letter read',
  S2: '2 letters read',
  S3: '3 letters read',
  S4: '1st digit read',
  S5: '2nd digit read',
  S6: '3rd digit read',
  G1: 'Group 1 (ends 1/2)',
  G2: 'Group 2 (ends 3/4)',
  G3: 'Group 3 (ends 5/6)',
  G4: 'Group 4 (ends 7/8)',
  G5: 'Group 5 (ends 9/0)',
  SR: 'Coding day',
  SN: 'Not a coding day',
  SC: 'S_Coding',
  SA: 'S_Allowed',
  SX: 'Trap (rejected)',
});

// ---------------------------------------------------------------------------
// Alphabet (Σ)
// ---------------------------------------------------------------------------
export const DAY_SYMBOLS = Object.freeze(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Hol']);
export const TIME_SYMBOLS = Object.freeze(['p', 'o']);

const isLetter = (s) => typeof s === 'string' && /^[A-Z]$/.test(s);
const isDigit = (s) => typeof s === 'string' && /^[0-9]$/.test(s);
const isDay = (s) => DAY_SYMBOLS.includes(s);
const isTime = (s) => TIME_SYMBOLS.includes(s);

/** Returns the symbol class used on the diagram edges: 'L', 'D', 'Day', 'Time' or null. */
export function classifySymbol(symbol) {
  if (isLetter(symbol)) return 'L';
  if (isDigit(symbol)) return 'D';
  if (isDay(symbol)) return 'Day';
  if (isTime(symbol)) return 'Time';
  return null; // not in Σ
}

// ---------------------------------------------------------------------------
// Coding-day groups (fourth / final plate digit → group → restricted day)
// ---------------------------------------------------------------------------
export const DIGIT_TO_GROUP = Object.freeze({
  1: STATES.G1, 2: STATES.G1,
  3: STATES.G2, 4: STATES.G2,
  5: STATES.G3, 6: STATES.G3,
  7: STATES.G4, 8: STATES.G4,
  9: STATES.G5, 0: STATES.G5,
});

export const GROUP_CODING_DAY = Object.freeze({
  G1: 'Mon',
  G2: 'Tue',
  G3: 'Wed',
  G4: 'Thu',
  G5: 'Fri',
});

// ---------------------------------------------------------------------------
// Transition function δ : Q × Σ → Q
// ---------------------------------------------------------------------------
/**
 * δ(state, symbol). Every pair not drawn in Figure 1 goes to SX,
 * and SX is absorbing (δ(SX, a) = SX for all a).
 */
export function transition(state, symbol) {
  switch (state) {
    // --- Plate letters: S0 -L-> S1 -L-> S2 -L-> S3 ---------------------------
    case STATES.S0:
      return isLetter(symbol) ? STATES.S1 : STATES.SX;
    case STATES.S1:
      return isLetter(symbol) ? STATES.S2 : STATES.SX;
    case STATES.S2:
      return isLetter(symbol) ? STATES.S3 : STATES.SX;

    // --- Plate digits: S3 -D-> S4 -D-> S5 -D-> S6 ----------------------------
    case STATES.S3:
      return isDigit(symbol) ? STATES.S4 : STATES.SX;
    case STATES.S4:
      return isDigit(symbol) ? STATES.S5 : STATES.SX;
    case STATES.S5:
      return isDigit(symbol) ? STATES.S6 : STATES.SX;

    // --- Fourth (final) digit picks the coding-day group ---------------------
    case STATES.S6:
      return isDigit(symbol) ? DIGIT_TO_GROUP[symbol] : STATES.SX;

    // --- Day symbol: coding day → SR, any other day → SN ---------------------
    case STATES.G1:
    case STATES.G2:
    case STATES.G3:
    case STATES.G4:
    case STATES.G5:
      if (!isDay(symbol)) return STATES.SX;
      return symbol === GROUP_CODING_DAY[state] ? STATES.SR : STATES.SN;

    // --- Time symbol ----------------------------------------------------------
    case STATES.SR: // coding day: peak → S_Coding, off-peak → S_Allowed
      if (symbol === 'p') return STATES.SC;
      if (symbol === 'o') return STATES.SA;
      return STATES.SX;
    case STATES.SN: // not a coding day: p, o → S_Allowed
      return isTime(symbol) ? STATES.SA : STATES.SX;

    // --- Terminal states: the string must end here ---------------------------
    case STATES.SC:
    case STATES.SA:
    case STATES.SX:
    default:
      return STATES.SX;
  }
}

// ---------------------------------------------------------------------------
// What each state expects next (used for rejection messages)
// ---------------------------------------------------------------------------
const EXPECTATION = Object.freeze({
  S0: 'a letter (1st plate letter)',
  S1: 'a letter (2nd plate letter)',
  S2: 'a letter (3rd plate letter)',
  S3: 'a digit (1st plate digit)',
  S4: 'a digit (2nd plate digit)',
  S5: 'a digit (3rd plate digit)',
  S6: 'a digit (4th plate digit)',
  G1: 'a day symbol (Mon–Sun or Hol)',
  G2: 'a day symbol (Mon–Sun or Hol)',
  G3: 'a day symbol (Mon–Sun or Hol)',
  G4: 'a day symbol (Mon–Sun or Hol)',
  G5: 'a day symbol (Mon–Sun or Hol)',
  SR: "a time symbol ('p' or 'o')",
  SN: "a time symbol ('p' or 'o')",
  SC: 'end of input',
  SA: 'end of input',
});

export const VERDICT = Object.freeze({
  CODING: 'CODING_RESTRICTION',
  ALLOWED: 'ALLOWED',
  REJECTED: 'REJECTED',
});

// ---------------------------------------------------------------------------
// Acceptor routine
// ---------------------------------------------------------------------------
/**
 * Runs M on an array of symbols, starting at S0.
 *
 * @param {string[]} symbols  standardized input, e.g. ['N','A','B','1','2','3','4','Mon','p']
 * @returns {{
 *   accepted: boolean,          // final state ∈ F
 *   finalState: string,
 *   verdict: 'CODING_RESTRICTION' | 'ALLOWED' | 'REJECTED',
 *   path: string[],             // S0, q1, q2, ... (length = symbols.length + 1)
 *   trace: {step:number, from:string, symbol:string, symbolClass:string|null, to:string}[],
 *   group: string|null,         // G1..G5 if reached
 *   codingDay: string|null,     // restricted day of that group
 *   reason: string              // explanation for the UI
 * }}
 */
export function runDFA(symbols) {
  const input = Array.isArray(symbols) ? symbols : [];
  let state = START_STATE;
  const path = [state];
  const trace = [];
  let group = null;
  let trapInfo = null;

  input.forEach((symbol, i) => {
    const from = state;
    const to = transition(from, symbol);
    trace.push({ step: i + 1, from, symbol, symbolClass: classifySymbol(symbol), to });

    if (to === STATES.SX && from !== STATES.SX && !trapInfo) {
      trapInfo = { from, symbol, position: i + 1 };
    }
    if (GROUP_CODING_DAY[to]) group = to;

    state = to;
    path.push(state);
  });

  const accepted = ACCEPTING_STATES.includes(state);
  let verdict = VERDICT.REJECTED;
  let reason;

  if (state === STATES.SC) {
    verdict = VERDICT.CODING;
    reason = `Plate is in ${group} (coding day: ${GROUP_CODING_DAY[group]}) and the time is peak hours. Coding restriction applies.`;
  } else if (state === STATES.SA) {
    verdict = VERDICT.ALLOWED;
    const day = input[input.length - 2];
    const time = input[input.length - 1];
    if (path[path.length - 2] === STATES.SR) {
      reason = `Today is the coding day for ${group}, but the time is off-peak ('${time}'). Vehicle is allowed.`;
    } else {
      reason = `${day} is not the coding day for ${group} (${GROUP_CODING_DAY[group]}). Vehicle is allowed.`;
    }
  } else if (trapInfo) {
    const shown = classifySymbol(trapInfo.symbol) === null ? `"${trapInfo.symbol}" (not in Σ)` : `"${trapInfo.symbol}"`;
    reason = `Trapped at symbol #${trapInfo.position}: state ${trapInfo.from} expected ${EXPECTATION[trapInfo.from]} but read ${shown}.`;
  } else {
    reason = `Input ended in ${state}, which is not an accepting state. Expected ${EXPECTATION[state] || 'more input'}.`;
  }

  return {
    accepted,
    finalState: state,
    verdict,
    path,
    trace,
    group,
    codingDay: group ? GROUP_CODING_DAY[group] : null,
    reason,
  };
}

export default runDFA;
