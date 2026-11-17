/*
 * ============================================================================
 *  PlateTextRecognizerModule.kt — Google ML Kit text recognition (Android)
 * ============================================================================
 *  Project : Design and Verification of a Regular Language Parser for
 *            MMDA Coding Rules Using Finite State Automata
 *  Course  : COSC 302 – Automata and Language Theory
 *  Group 12: BULAN, ZALDY MARCUS, B.
 *            Del Mundo, M. A.
 *            Librado, C. T.
 *            Paz, L. H.
 *
 *  Reads a photo with ML Kit's on-device Latin text recognizer and returns
 *  every recognized line (top to bottom). Standardization and the DFA run in JS.
 * ============================================================================
 */
package expo.modules.platetextrecognizer

import android.net.Uri
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import expo.modules.kotlin.Promise
import expo.modules.kotlin.exception.CodedException
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import java.io.File

class PlateTextRecognizerModule : Module() {
  private val recognizer by lazy { TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS) }

  override fun definition() = ModuleDefinition {
    Name("PlateTextRecognizer")

    AsyncFunction("recognize") { uriString: String, promise: Promise ->
      try {
        val context = appContext.reactContext
          ?: throw IllegalStateException("React context is not available")

        val uri: Uri = when {
          uriString.startsWith("content://") || uriString.startsWith("file://") -> Uri.parse(uriString)
          else -> Uri.fromFile(File(uriString))
        }

        val image = InputImage.fromFilePath(context, uri)
        recognizer.process(image)
          .addOnSuccessListener { result ->
            val lines = result.textBlocks
              .flatMap { block -> block.lines }
              .sortedBy { line -> line.boundingBox?.top ?: 0 }
              .map { line -> line.text }
            promise.resolve(lines)
          }
          .addOnFailureListener { error ->
            promise.reject(CodedException("ERR_TEXT_RECOGNITION", error.message ?: "Text recognition failed", error))
          }
      } catch (error: Exception) {
        promise.reject(CodedException("ERR_TEXT_RECOGNITION", error.message ?: "Could not read the image", error))
      }
    }
  }
}
