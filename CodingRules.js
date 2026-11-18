/**
 * ============================================================================
 *  CodingRules.js — LGU coding hours and the national holiday calendar
 * ============================================================================
 *  Both tables feed the PREPROCESSING layer only. They decide which symbols
 *  are handed to M (DFAEngine.js); they never change Q, Σ or δ:
 *
 *    • City rules  → which clock times count as peak, i.e. Time symbol 'p' / 'o'
 *    • Holidays    → whether the Day symbol is 'Hol' instead of Mon … Sun
 *
 *  So M stays exactly as drawn in Figure 1 for every city and every date.
 * ============================================================================
 */

// ---------------------------------------------------------------------------
//  LGU coding hours
// ---------------------------------------------------------------------------
/** Windows are minutes from midnight, both ends inclusive (10:00 is still peak, 10:01 is not). */
export const MMDA_WINDOWS = Object.freeze([
  { start: 7 * 60, end: 10 * 60, label: 'Morning' },
  { start: 17 * 60, end: 20 * 60, label: 'Evening' },
]);

const DEFAULT_RULE = Object.freeze({
  windows: MMDA_WINDOWS,
  source: 'MMDA UVVRP',
  note: null,
});

/**
 * Cities whose coding hours differ from the MMDA default. Every city not listed
 * here follows MMDA_WINDOWS.
 */
export const CITY_RULES = Object.freeze({
  Makati: Object.freeze({
    windows: Object.freeze([{ start: 7 * 60, end: 19 * 60, label: 'All day' }]),
    source: 'Makati City coding ordinance',
    note: 'Makati has no window hours: coding runs 7 AM–7 PM straight.',
  }),
});

/** Coding rule for a city (falls back to the MMDA default). */
export function rulesFor(city) {
  return CITY_RULES[city] || DEFAULT_RULE;
}

/** 'p' if the time falls inside one of the city's coding windows, else 'o'. */
export function timeToSymbol(hours24, minutes, city) {
  const t = hours24 * 60 + minutes;
  return rulesFor(city).windows.some((w) => t >= w.start && t <= w.end) ? 'p' : 'o';
}

const clock = (mins, withMer = true) => {
  const h24 = Math.floor(mins / 60);
  const m = mins % 60;
  const h = h24 % 12 === 0 ? 12 : h24 % 12;
  const text = m ? `${h}:${String(m).padStart(2, '0')}` : `${h}`;
  return withMer ? `${text} ${h24 >= 12 ? 'PM' : 'AM'}` : text;
};

/** "7:00 AM"-style label for a minute-of-day. */
export const clockLabel = (mins) => {
  const h24 = Math.floor(mins / 60);
  const h = h24 % 12 === 0 ? 12 : h24 % 12;
  return `${h}:${String(mins % 60).padStart(2, '0')} ${h24 >= 12 ? 'PM' : 'AM'}`;
};

/** Short form of one window: "7–10 AM", "5–8 PM", "7 AM–7 PM". */
export function windowShort(w) {
  const sameMer = (w.start >= 12 * 60) === (w.end >= 12 * 60);
  return sameMer ? `${clock(w.start, false)}–${clock(w.end)}` : `${clock(w.start)}–${clock(w.end)}`;
}

/** All of a city's windows, e.g. "7–10 AM · 5–8 PM". */
export const hoursShort = (city, sep = ' · ') => rulesFor(city).windows.map(windowShort).join(sep);

/**
 * Where a given minute-of-day sits in the city's coding hours.
 * @returns {{ inside: object|null, next: object|null }}  the window containing t, and the next one after t
 */
export function windowAt(city, t) {
  const ws = rulesFor(city).windows;
  return {
    inside: ws.find((w) => t >= w.start && t <= w.end) || null,
    next: ws.find((w) => w.start > t) || null,
  };
}

// ---------------------------------------------------------------------------
//  National holidays (coding is lifted on regular and special non-working days)
// ---------------------------------------------------------------------------
/**
 * Sources:
 *   2026 — Proclamation No. 1006 (s. 2025)
 *   2027 — Proclamation No. 1427 (s. 2026), signed 8 Sep 2026
 * Special WORKING days (EDSA anniversary, Feb 25) are not holidays and are left out.
 * Eid'l Fitr and Eid'l Adha are proclaimed separately each year once their dates
 * are set, so they are not listed; use the manual "Today is a holiday" switch.
 */
export const HOLIDAYS = Object.freeze({
  // ---- 2026 ----
  '2026-01-01': "New Year's Day",
  '2026-02-17': 'Chinese New Year',
  '2026-04-02': 'Maundy Thursday',
  '2026-04-03': 'Good Friday',
  '2026-04-04': 'Black Saturday',
  '2026-04-09': 'Araw ng Kagitingan',
  '2026-05-01': 'Labor Day',
  '2026-06-12': 'Independence Day',
  '2026-08-21': 'Ninoy Aquino Day',
  '2026-08-31': 'National Heroes Day',
  '2026-11-01': "All Saints' Day",
  '2026-11-02': "All Souls' Day",
  '2026-11-30': 'Bonifacio Day',
  '2026-12-08': 'Immaculate Conception',
  '2026-12-24': 'Christmas Eve',
  '2026-12-25': 'Christmas Day',
  '2026-12-30': 'Rizal Day',
  '2026-12-31': 'Last Day of the Year',
  // ---- 2027 ----
  '2027-01-01': "New Year's Day",
  '2027-02-06': 'Chinese New Year',
  '2027-03-25': 'Maundy Thursday',
  '2027-03-26': 'Good Friday',
  '2027-03-27': 'Black Saturday',
  '2027-04-09': 'Araw ng Kagitingan',
  '2027-05-01': 'Labor Day',
  '2027-06-12': 'Independence Day',
  '2027-08-21': 'Ninoy Aquino Day',
  '2027-08-30': 'National Heroes Day',
  '2027-11-01': "All Saints' Day",
  '2027-11-02': "All Souls' Day",
  '2027-11-30': 'Bonifacio Day',
  '2027-12-08': 'Immaculate Conception',
  '2027-12-24': 'Christmas Eve',
  '2027-12-25': 'Christmas Day',
  '2027-12-30': 'Rizal Day',
  '2027-12-31': 'Last Day of the Year',
});

/** Last year the calendar covers; after it, only the manual switch works. */
export const HOLIDAYS_THROUGH = 2027;

/** Local-date key "YYYY-MM-DD" (the phone's own calendar day, not UTC). */
export function dateKey(d) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

/** Name of the national holiday on that date, or null. */
export function holidayOn(date) {
  return HOLIDAYS[dateKey(date)] || null;
}

/** First holiday on or after `from`, as { date, name }, or null if the calendar has run out. */
export function nextHoliday(from) {
  const key = dateKey(from);
  const hit = Object.keys(HOLIDAYS).sort().find((k) => k >= key);
  if (!hit) return null;
  const [y, m, d] = hit.split('-').map(Number);
  return { date: new Date(y, m - 1, d), name: HOLIDAYS[hit] };
}

const JS_DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

/** The next date (today included) that falls on a given day symbol, e.g. 'Thu'. */
export function nextDateFor(daySymbol, from = new Date()) {
  const target = JS_DAYS.indexOf(daySymbol);
  const d = new Date(from.getFullYear(), from.getMonth(), from.getDate());
  if (target < 0) return d;
  d.setDate(d.getDate() + ((target - d.getDay() + 7) % 7));
  return d;
}
