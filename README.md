<!--
  MMDA Coding DFA Parser — COSC 302 Automata and Language Theory
  Group 12: BULAN, ZALDY MARCUS, B. · Del Mundo, M. A. · Librado, C. T. · Paz, L. H.
-->
# PlateWise PH — MMDA Coding DFA Parser (React Native + Expo, Android)

| File | Role |
|---|---|
| `DFAEngine.js` | Pure automaton M = (Q, Σ, δ, S0, F). No UI. `runDFA(symbols)` returns final state, verdict, path and trace. |
| `PlateOCR.js` | Main OCR: Google ML Kit text recognition on the phone. Crops the photo to the scan frame first. |
| `modules/plate-text-recognizer/` | Local Expo module (Kotlin) wrapping ML Kit's bundled Latin text recognizer. |
| `App.js` | PlateWise UI (splash, onboarding, check plate, scan, result, schedule, guide) and the preprocessing layer: `standardizeInput()`, day/time → symbol. |
| My cars | Named cars saved on the phone (AsyncStorage). The My cars tab shows each car's live coding status (DFA run every 30 s); add, rename, remove. Saved cars also appear as chips on Check plate / Schedule. |
| `assets/` | PlateWise logo: app icon, adaptive icon, splash, header mark. |

## Try it on your phone right away (Expo Go)
```bash
npm install
npm start          # scan the QR code with the Expo Go app (same Wi-Fi)
```

## Installable APK
```bash
npx eas-cli@latest login
npm run build:apk  # open the link/QR it prints on your phone, then install
```

In the APK, plates are read by ML Kit on the phone (offline; the model ships inside the APK).
Plate scanning only works in the installed APK (ML Kit isn't in Expo Go); in Expo Go you can still type plates.

## Input string
`L L L D D D D Day Time` → e.g. `[N, A, B, 1, 2, 3, 2, Mon, p]` → S0 S1 S2 S3 S4 S5 S6 G1 SR **SC**
