/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
/* tslint:disable */
// Copyright 2024 Google LLC

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     https://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {GoogleGenAI} from '@google/genai';
import {
  Detection,
  FaceDetector,
  FilesetResolver,
  HandLandmarker,
  NormalizedLandmark,
  ObjectDetector,
  PoseLandmarker,
} from '@mediapipe/tasks-vision';
import {useEffect, useRef, useState} from 'react';

// Simplified the vision task type for this feature
type DetectionMode = 'object' | 'person_analysis';

interface BoundingBox {
  originX: number;
  originY: number;
  width: number;
  height: number;
}

const ai = new GoogleGenAI({apiKey: process.env.API_KEY});

let lastVideoTime = -1;
let requestAnimationId: number;

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // MediaPipe model refs
  const objectDetector = useRef<ObjectDetector | null>(null);
  const faceDetector = useRef<FaceDetector | null>(null);
  const handLandmarker = useRef<HandLandmarker | null>(null); // Kept for potential future use
  const poseLandmarker = useRef<PoseLandmarker | null>(null); // Kept for potential future use
  const vision = useRef<FilesetResolver | null>(null);

  // App state
  const [inputPrompt, setInputPrompt] = useState<string>('age and gender');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [loadingMessage, setLoadingMessage] = useState<string>(
    'Initializing AI...',
  );
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [detectionMode, setDetectionMode] =
    useState<DetectionMode>('person_analysis');
  const [detectionTarget, setDetectionTarget] = useState<string>('');

  // Person analysis state
  const [personAnalysisResults, setPersonAnalysisResults] = useState<Array<{
    bbox: BoundingBox;
    text: string;
  }> | null>(null);
  const lastAnalysisTime = useRef<number>(0);
  const isAnalyzing = useRef<boolean>(false);

  // Main setup effect
  useEffect(() => {
    async function setup() {
      try {
        setLoadingMessage('Initializing vision tasks...');
        vision.current = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
        );
        await setupCamera();
        // Load the initial model
        await switchModel('person_analysis');
      } catch (e) {
        console.error('Failed during setup', e);
        setErrorMessage('Failed to initialize. Please refresh the page.');
        setIsLoading(false);
      }
    }
    setup();

    // Cleanup
    return () => {
      window.cancelAnimationFrame(requestAnimationId);
      videoRef.current?.srcObject
        ? (videoRef.current.srcObject as MediaStream)
            .getTracks()
            .forEach((track) => track.stop())
        : null;
      objectDetector.current?.close();
      faceDetector.current?.close();
      handLandmarker.current?.close();
      poseLandmarker.current?.close();
    };
  }, []);

  async function switchModel(task: DetectionMode) {
    setIsLoading(true);
    setLoadingMessage('Loading AI vision model...');
    // Close active models
    objectDetector.current?.close();
    faceDetector.current?.close();
    objectDetector.current = null;
    faceDetector.current = null;

    try {
      if (task === 'object') {
        objectDetector.current = await ObjectDetector.createFromOptions(
          vision.current! as any,
          {
            baseOptions: {
              modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`,
              delegate: 'GPU',
            },
            scoreThreshold: 0.5,
            runningMode: 'VIDEO',
          },
        );
      } else if (task === 'person_analysis') {
        faceDetector.current = await FaceDetector.createFromOptions(
          vision.current! as any,
          {
            baseOptions: {
              modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
          },
        );
      }
      setIsLoading(false);
    } catch (e) {
      console.error('Failed to switch model', e);
      setErrorMessage(
        'Failed to load the selected AI model. Please try another prompt.',
      );
      setIsLoading(false);
    }
  }

  async function setupCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMessage('Camera access is not supported by this browser.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {facingMode: 'user'},
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.addEventListener('loadeddata', predict);
      }
    } catch (e) {
      console.error('Failed to get camera feed', e);
      setErrorMessage(
        'Could not access the camera. Please grant permission and refresh.',
      );
    }
  }

  async function analyzeAllFaces(detections: Detection[]) {
    if (
      !detections ||
      detections.length === 0 ||
      !videoRef.current ||
      isAnalyzing.current
    )
      return;
    isAnalyzing.current = true;

    try {
      const analysisPromises = detections.map(async (detection) => {
        if (!detection.boundingBox) return null;

        const bbox = detection.boundingBox!;

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = bbox.width;
        tempCanvas.height = bbox.height;
        const tempCtx = tempCanvas.getContext('2d')!;
        tempCtx.drawImage(
          videoRef.current!,
          bbox.originX,
          bbox.originY,
          bbox.width,
          bbox.height,
          0,
          0,
          bbox.width,
          bbox.height,
        );

        const imageDataUrl = tempCanvas.toDataURL('image/jpeg');
        const base64Data = imageDataUrl.split(',')[1];

        const prompt = `Analyze the person in the image based on this user query: "${inputPrompt}". Provide a concise, single-line summary of your analysis. Include estimated age and gender if not specified. Format as a comma-separated list of key-value pairs (e.g., Gender: Female, Age: 25-30).`;

        const response = await ai.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: {
            parts: [
              {inlineData: {mimeType: 'image/jpeg', data: base64Data}},
              {text: prompt},
            ],
          },
        });

        return {
          bbox: {
            originX: bbox.originX,
            originY: bbox.originY,
            width: bbox.width,
            height: bbox.height,
          },
          text: response.text,
        };
      });

      const results = (await Promise.all(analysisPromises)).filter(Boolean);
      setPersonAnalysisResults(
        results as Array<{bbox: BoundingBox; text: string}>,
      );
    } catch (e) {
      console.error('Failed to analyze faces', e);
    } finally {
      isAnalyzing.current = false;
    }
  }

  function predict() {
    const video = videoRef.current;
    if (!video || video.paused || video.ended) return;
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const renderLoop = () => {
      if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const nowMs = performance.now();
        let results: any;
        if (detectionMode === 'object' && objectDetector.current) {
          results = objectDetector.current.detectForVideo(video, nowMs);
        } else if (
          detectionMode === 'person_analysis' &&
          faceDetector.current
        ) {
          results = faceDetector.current.detectForVideo(video, nowMs);
        }

        if (
          detectionMode === 'person_analysis' &&
          results?.detections?.length > 0
        ) {
          if (nowMs - lastAnalysisTime.current > 3000) {
            lastAnalysisTime.current = nowMs;
            analyzeAllFaces(results.detections);
          }
        }
        drawResults(results, ctx, canvas.width, canvas.height);
      }
      requestAnimationId = window.requestAnimationFrame(renderLoop);
    };
    renderLoop();
  }

  function drawResults(
    results: any,
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
  ) {
    ctx.clearRect(0, 0, width, height);
    if (!results) return;

    if (detectionMode === 'object')
      drawObjectDetections(results.detections, ctx, width);
    if (detectionMode === 'person_analysis')
      drawFaceDetections(results.detections, ctx, width);
  }

  function drawObjectDetections(
    detections: Detection[],
    ctx: CanvasRenderingContext2D,
    width: number,
  ) {
    const targets = detectionTarget
      .toLowerCase()
      .trim()
      .split(/[\s,]+|and/i)
      .filter(Boolean);

    for (const detection of detections) {
      if (!detection.boundingBox || !detection.categories.length) continue;
      const category = detection.categories[0].categoryName.toLowerCase();
      if (targets.length > 0) {
        const matches = targets.some((target) => category.includes(target));
        if (!matches) {
          continue;
        }
      }

      const bbox = detection.boundingBox!;
      const mirroredX = width - bbox.originX - bbox.width;

      ctx.strokeStyle = '#3B68FF';
      ctx.lineWidth = 4;
      ctx.strokeRect(mirroredX, bbox.originY, bbox.width, bbox.height);
      const label = `${category} (${Math.round(
        detection.categories[0].score * 100,
      )}%)`;
      drawLabel(label, mirroredX, bbox.originY, ctx);
    }
  }

  function drawFaceDetections(
    detections: Detection[],
    ctx: CanvasRenderingContext2D,
    width: number,
  ) {
    if (!detections) return;

    // First, draw a bounding box for every detected face
    for (const detection of detections) {
      if (!detection.boundingBox) continue;
      const bbox = detection.boundingBox!;
      const mirroredX = width - bbox.originX - bbox.width;
      ctx.strokeStyle = '#3B68FF';
      ctx.lineWidth = 2;
      ctx.strokeRect(mirroredX, bbox.originY, bbox.width, bbox.height);
    }

    // Then, try to attach analysis labels to the closest faces
    if (personAnalysisResults && personAnalysisResults.length > 0) {
      const unattachedDetections = [...detections];

      for (const result of personAnalysisResults) {
        let bestMatch: Detection | null = null;
        let bestMatchIndex = -1;
        let minDistance = Infinity;

        const resultCenter = {
          x: result.bbox.originX + result.bbox.width / 2,
          y: result.bbox.originY + result.bbox.height / 2,
        };

        for (let i = 0; i < unattachedDetections.length; i++) {
          const detection = unattachedDetections[i];
          if (!detection.boundingBox) continue;
          const detBbox = detection.boundingBox!;
          const detCenter = {
            x: detBbox.originX + detBbox.width / 2,
            y: detBbox.originY + detBbox.height / 2,
          };
          const distance = Math.sqrt(
            Math.pow(resultCenter.x - detCenter.x, 2) +
              Math.pow(resultCenter.y - detCenter.y, 2),
          );
          if (distance < minDistance) {
            minDistance = distance;
            bestMatch = detection;
            bestMatchIndex = i;
          }
        }

        // Heuristic: If a reasonably close match is found, draw the label and "claim" the detection
        if (bestMatch && minDistance < bestMatch.boundingBox!.width * 0.75) {
          const bbox = bestMatch.boundingBox!;
          const mirroredX = width - bbox.originX - bbox.width;
          drawLabel(
            result.text,
            mirroredX,
            bbox.originY + bbox.height,
            ctx,
            '#3B68FF',
            'bottom',
          );

          // Remove from pool of available detections so it can't be labeled twice
          unattachedDetections.splice(bestMatchIndex, 1);
        }
      }
    }
  }

  function drawLandmarks(
    landmarks: NormalizedLandmark[][],
    connections: {start: number; end: number}[],
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    color: string,
  ) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    for (const landmark of landmarks) {
      for (const connection of connections) {
        const start = landmark[connection.start];
        const end = landmark[connection.end];
        if (start && end) {
          const mirroredStartX = width - start.x * width;
          const mirroredEndX = width - end.x * width;
          ctx.beginPath();
          ctx.moveTo(mirroredStartX, start.y * height);
          ctx.lineTo(mirroredEndX, end.y * height);
          ctx.stroke();
        }
      }
    }
  }

  function drawLabel(
    label: string,
    x: number,
    y: number,
    ctx: CanvasRenderingContext2D,
    color = '#3B68FF',
    position: 'top' | 'bottom' = 'top',
  ) {
    ctx.font = '16px Space Mono';
    const lines = label.split(', ');
    let maxWidth = 0;
    lines.forEach((line) => {
      const width = ctx.measureText(line).width;
      if (width > maxWidth) {
        maxWidth = width;
      }
    });

    const lineHeight = 20;
    const bgHeight = lines.length * lineHeight + 8;
    const bgWidth = maxWidth + 16;
    let bgX = x;
    let bgY;

    if (position === 'top') {
      bgY = y > bgHeight ? y - bgHeight : y;
    } else {
      bgY = y;
    }

    ctx.fillStyle = color;
    ctx.fillRect(bgX, bgY, bgWidth, bgHeight);
    ctx.fillStyle = color === '#FFFF00' ? '#000000' : '#FFFFFF';
    ctx.textBaseline = 'top';
    lines.forEach((line, index) => {
      ctx.fillText(line, bgX + 8, bgY + 4 + index * lineHeight);
    });
  }

  async function handleModeChange(mode: DetectionMode) {
    if (mode === detectionMode) return;

    setDetectionMode(mode);
    setPersonAnalysisResults(null);
    setErrorMessage('');

    if (mode === 'object') {
      const newPrompt = 'cup'; // A sensible default
      setInputPrompt(newPrompt);
      setDetectionTarget(newPrompt);
    } else {
      // person_analysis
      const newPrompt = 'age and gender'; // A sensible default
      setInputPrompt(newPrompt);
      setDetectionTarget('');
    }

    await switchModel(mode);
  }

  async function handlePromptSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!inputPrompt) return;

    if (detectionMode === 'object') {
      setDetectionTarget(inputPrompt);
    } else if (detectionMode === 'person_analysis') {
      // The prompt has changed, so we want to re-analyze immediately.
      lastAnalysisTime.current = 0;
      setPersonAnalysisResults(null); // Clear previous results
    }
  }

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-black">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 min-w-full min-h-full object-cover"
        style={{transform: 'scaleX(-1)'}}
      />
      <canvas
        ref={canvasRef}
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 min-w-full min-h-full"
      />

      {(isLoading || errorMessage) && (
        <div
          className={`absolute inset-0 z-20 flex flex-col items-center justify-center text-white ${
            errorMessage
              ? 'bg-red-800 bg-opacity-90'
              : 'bg-black bg-opacity-70'
          }`}>
          {errorMessage ? (
            <p className="p-4 text-center text-lg">{errorMessage}</p>
          ) : (
            <>
              <svg
                className="-ml-1 mr-3 h-10 w-10 animate-spin text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <p className="mt-2 text-lg">{loadingMessage}</p>
            </>
          )}
        </div>
      )}

      <div className="absolute bottom-5 left-1/2 z-10 w-full max-w-lg -translate-x-1/2 px-4">
        <div className="mb-2 flex justify-center gap-2">
          <button
            onClick={() => handleModeChange('person_analysis')}
            disabled={isLoading}
            className={`rounded-full px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50 ${
              detectionMode === 'person_analysis'
                ? 'bg-[#3B68FF] text-white'
                : 'bg-black bg-opacity-50 text-gray-300 hover:bg-opacity-70'
            }`}>
            Analyze Person
          </button>
          <button
            onClick={() => handleModeChange('object')}
            disabled={isLoading}
            className={`rounded-full px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50 ${
              detectionMode === 'object'
                ? 'bg-[#3B68FF] text-white'
                : 'bg-black bg-opacity-50 text-gray-300 hover:bg-opacity-70'
            }`}>
            Detect Object
          </button>
        </div>
        <form
          onSubmit={handlePromptSubmit}
          className="flex items-center gap-2 rounded-full border border-gray-600 bg-black bg-opacity-50 p-2 pl-4">
          <input
            type="text"
            value={inputPrompt}
            onChange={(e) => setInputPrompt(e.target.value)}
            placeholder={
              detectionMode === 'person_analysis'
                ? 'Ask for more details (e.g., "mood")'
                : 'Describe what to track...'
            }
            className="w-full flex-grow bg-transparent text-white placeholder-gray-400 focus:outline-none"
            disabled={isLoading || !!errorMessage}
            aria-label="Object to track"
          />
          <button
            type="submit"
            className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-[#3B68FF] text-white transition-opacity disabled:opacity-50"
            disabled={isLoading || !!errorMessage}
            aria-label="Update Tracker">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2.5}
              stroke="currentColor"
              className="h-6 w-6">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 19.5v-15m0 0l-6.75 6.75m6.75-6.75l6.75 6.75"
              />
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
