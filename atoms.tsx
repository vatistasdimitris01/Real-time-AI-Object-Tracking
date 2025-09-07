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

import {atom} from 'jotai';
import {DetectTypes} from './Types.tsx';

// Define interfaces for complex atom types
interface BoundingBox2D {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
}

interface BoundingBox3D {
  center: number[];
  size: number[];
  rpy: number[];
  label: string;
}

interface BoundingBoxMask {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  imageData: string;
}

interface Point {
  point: {
    x: number;
    y: number;
  };
  label: string;
}

type Line = [[number, number][], string];

// Atoms
export const ImageSrcAtom = atom<string | null>('assets/1.png');
export const IsUploadedImageAtom = atom(false);
export const BoundingBoxes2DAtom = atom<BoundingBox2D[]>([]);
export const BoundingBoxes3DAtom = atom<BoundingBox3D[]>([]);
export const BoundingBoxMasksAtom = atom<BoundingBoxMask[]>([]);
// Fix: The previous derived atom was causing type inference issues.
// A primitive atom is writable by default and serves the same purpose.
export const ShareStream = atom<MediaStream | null>(null);
export const DetectTypeAtom = atom<DetectTypes>('2D bounding boxes');
export const VideoRefAtom = atom({current: null as HTMLVideoElement | null});
export const FOVAtom = atom(90);
export const ImageSentAtom = atom(false);
export const PointsAtom = atom<Point[]>([]);
export const RevealOnHoverModeAtom = atom(false);
export const HoverEnteredAtom = atom(false);
export const HoveredBoxAtom = atom<number | null>(null);
export const DrawModeAtom = atom(false);
export const LinesAtom = atom<Line[]>([]);
export const ActiveColorAtom = atom('#3B68FF');
export const BumpSessionAtom = atom(0);
export const IsLoadingAtom = atom(false);
export const TemperatureAtom = atom(0.25);

const defaultPrompts: Record<DetectTypes, [string, string, string]> = {
  '2D bounding boxes': [
    'Detect',
    'items',
    ', with no more than 20 items. Output a json list where each entry contains the 2D bounding box in "box_2d" and a text label in "label".',
  ],
  'Segmentation masks': [
    'Segment',
    'everything',
    '. Output a json list where each entry contains the 2D bounding box in "box_2d", a text label in the key "label". Use descriptive labels.',
  ],
  Points: [
    'Detect key points on',
    'a person',
    ' in the image. Output a json list where each entry contains a point in "point" and a text label in "label".',
  ],
  '3D bounding boxes': [
    'Detect',
    'cars',
    '. Respond in a JSON list, with each entry having a 3D bounding box in "box_3d" and a text label in "label".',
  ],
};

export const PromptsAtom = atom(defaultPrompts);

export const CustomPromptsAtom = atom<Record<DetectTypes, string>>({
  '2D bounding boxes': '',
  'Segmentation masks': '',
  Points: '',
  '3D bounding boxes': '',
});
