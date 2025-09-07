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

type VisionTask = 'object' | 'face' | 'hands' | 'pose';

let lastVideoTime = -1;
let requestAnimationId: number;

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // MediaPipe model refs
  const objectDetector = useRef<ObjectDetector | null>(null);
  const faceDetector = useRef<FaceDetector | null>(null);
  const handLandmarker = useRef<HandLandmarker | null>(null);
  const poseLandmarker = useRef<PoseLandmarker | null>(null);
  const vision = useRef<FilesetResolver | null>(null);

  // App state
  const [inputPrompt, setInputPrompt] = useState<string>('person');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [loadingMessage, setLoadingMessage] = useState<string>(
    'Initializing AI...',
  );
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [activeTask, setActiveTask] = useState<VisionTask>('object');
  const [detectionTarget, setDetectionTarget] = useState<string>('person');

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
        await switchModel(activeTask);
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

  async function switchModel(task: VisionTask) {
    setIsLoading(true);
    setLoadingMessage('Loading AI vision model...');
    // Close all current models
    objectDetector.current?.close();
    faceDetector.current?.close();
    handLandmarker.current?.close();
    poseLandmarker.current?.close();
    objectDetector.current = null;
    faceDetector.current = null;
    handLandmarker.current = null;
    poseLandmarker.current = null;

    try {
      if (task === 'object') {
        // Fix: Cast vision.current to 'any' to resolve a type mismatch with MediaPipe's createFromOptions.
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
      } else if (task === 'face') {
        // Fix: Cast vision.current to 'any' to resolve a type mismatch with MediaPipe's createFromOptions.
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
      } else if (task === 'hands') {
        // Fix: Cast vision.current to 'any' to resolve a type mismatch with MediaPipe's createFromOptions.
        handLandmarker.current = await HandLandmarker.createFromOptions(
          vision.current! as any,
          {
            baseOptions: {
              modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.tflite`,
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numHands: 2,
          },
        );
      } else if (task === 'pose') {
        // Fix: Cast vision.current to 'any' to resolve a type mismatch with MediaPipe's createFromOptions.
        poseLandmarker.current = await PoseLandmarker.createFromOptions(
          vision.current! as any,
          {
            baseOptions: {
              modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.tflite`,
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numPoses: 2,
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
        const now = performance.now();
        let results: any;
        if (activeTask === 'object' && objectDetector.current) {
          results = objectDetector.current.detectForVideo(video, now);
        } else if (activeTask === 'face' && faceDetector.current) {
          results = faceDetector.current.detectForVideo(video, now);
        } else if (activeTask === 'hands' && handLandmarker.current) {
          results = handLandmarker.current.detectForVideo(video, now);
        } else if (activeTask === 'pose' && poseLandmarker.current) {
          results = poseLandmarker.current.detectForVideo(video, now);
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

    if (activeTask === 'object')
      drawObjectDetections(results.detections, ctx, width);
    if (activeTask === 'face')
      drawFaceDetections(results.detections, ctx, width);
    if (activeTask === 'hands')
      drawLandmarks(
        results.landmarks,
        HandLandmarker.HAND_CONNECTIONS,
        ctx,
        width,
        height,
        '#FF0000',
      );
    if (activeTask === 'pose')
      drawLandmarks(
        results.landmarks,
        PoseLandmarker.POSE_CONNECTIONS,
        ctx,
        width,
        height,
        '#00FF00',
      );
  }

  function drawObjectDetections(
    detections: Detection[],
    ctx: CanvasRenderingContext2D,
    width: number,
  ) {
    for (const detection of detections) {
      const category = detection.categories[0].categoryName.toLowerCase();
      if (
        detectionTarget &&
        !category.includes(detectionTarget.toLowerCase().trim())
      ) {
        continue;
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
    for (const detection of detections) {
      const bbox = detection.boundingBox!;
      const mirroredX = width - bbox.originX - bbox.width;
      ctx.strokeStyle = '#FFFF00';
      ctx.lineWidth = 2;
      ctx.strokeRect(mirroredX, bbox.originY, bbox.width, bbox.height);
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
  ) {
    ctx.font = '20px Space Mono';
    const textWidth = ctx.measureText(label).width;
    const bgHeight = 30;
    const bgWidth = textWidth + 16;
    let bgX = x;
    let bgY = y > bgHeight ? y - bgHeight : y;

    ctx.fillStyle = '#3B68FF';
    ctx.fillRect(bgX, bgY, bgWidth, bgHeight);
    ctx.fillStyle = '#FFFFFF';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, bgX + 8, bgY + bgHeight / 2);
  }

  async function handlePromptSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!inputPrompt) return;

    setIsLoading(true);
    setLoadingMessage('Updating tracker...');

    try {
      const lowerCasePrompt = inputPrompt.toLowerCase().trim();

      const faceKeywords = ['face', 'head', 'selfie', 'visage'];
      const handKeywords = ['hand', 'hands', 'finger', 'fingers', 'palm', 'wave'];
      const poseKeywords = ['pose', 'body', 'stand', 'dance', 'jump', 'posture'];

      let newActiveTask: VisionTask = 'object'; // Default to object
      let newDetectionTarget = lowerCasePrompt;

      if (faceKeywords.some((keyword) => lowerCasePrompt.includes(keyword))) {
        newActiveTask = 'face';
        newDetectionTarget = '';
      } else if (
        handKeywords.some((keyword) => lowerCasePrompt.includes(keyword))
      ) {
        newActiveTask = 'hands';
        newDetectionTarget = '';
      } else if (
        poseKeywords.some((keyword) => lowerCasePrompt.includes(keyword))
      ) {
        newActiveTask = 'pose';
        newDetectionTarget = '';
      }

      setDetectionTarget(newDetectionTarget);

      if (newActiveTask !== activeTask) {
        setActiveTask(newActiveTask);
        await switchModel(newActiveTask);
      }
    } catch (error) {
      console.error('Error switching model:', error);
      setErrorMessage('Could not update tracker. Please try again.');
    } finally {
      setIsLoading(false);
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
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"
      />

      {(isLoading || errorMessage) && (
        <div
          className={`absolute inset-0 z-20 flex flex-col items-center justify-center text-white ${errorMessage ? 'bg-red-800 bg-opacity-90' : 'bg-black bg-opacity-70'}`}>
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
        <form
          onSubmit={handlePromptSubmit}
          className="flex gap-2 rounded-full border border-gray-600 bg-black bg-opacity-50 p-2">
          <input
            type="text"
            value={inputPrompt}
            onChange={(e) => setInputPrompt(e.target.value)}
            placeholder="Describe what to track..."
            className="w-full flex-grow bg-transparent px-4 text-white placeholder-gray-400 focus:outline-none"
            disabled={isLoading || !!errorMessage}
            aria-label="Object to track"
          />
          <button
            type="submit"
            className="rounded-full bg-[#3B68FF] px-6 py-2 text-white disabled:opacity-50"
            disabled={isLoading || !!errorMessage}>
            Update
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
