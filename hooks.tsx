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

import {useSetAtom} from 'jotai';
import {
  BoundingBoxes2DAtom,
  BoundingBoxes3DAtom,
  BoundingBoxMasksAtom,
  DrawModeAtom,
  ImageSentAtom,
  ImageSrcAtom,
  IsUploadedImageAtom,
  LinesAtom,
  PointsAtom,
  ShareStream,
} from './atoms.tsx';

export function useResetState() {
  const setImageSrc = useSetAtom(ImageSrcAtom);
  const setBoundingBoxes2D = useSetAtom(BoundingBoxes2DAtom);
  const setBoundingBoxes3D = useSetAtom(BoundingBoxes3DAtom);
  const setBoundingBoxMasks = useSetAtom(BoundingBoxMasksAtom);
  const setPoints = useSetAtom(PointsAtom);
  const setLines = useSetAtom(LinesAtom);
  const setDrawMode = useSetAtom(DrawModeAtom);
  const setStream = useSetAtom(ShareStream);
  const setIsUploadedImage = useSetAtom(IsUploadedImageAtom);
  const setImageSent = useSetAtom(ImageSentAtom);

  return () => {
    setImageSrc(null);
    setBoundingBoxes2D([]);
    setBoundingBoxes3D([]);
    setBoundingBoxMasks([]);
    setPoints([]);
    setLines([]);
    setDrawMode(false);
    setStream((prev) => {
      prev?.getTracks().forEach((track) => track.stop());
      return null;
    });
    setIsUploadedImage(false);
    setImageSent(false);
  };
}
