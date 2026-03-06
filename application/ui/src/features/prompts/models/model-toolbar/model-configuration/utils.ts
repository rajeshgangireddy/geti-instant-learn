/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { MatcherModel, ModelType, PerDINOModel, SoftMatcherModel, YoloeModel } from '@/api';

export const isMatcherModel = (m: ModelType): m is MatcherModel => m.config.model_type === 'matcher';
export const isPerDINOModel = (m: ModelType): m is PerDINOModel => m.config.model_type === 'perdino';
export const isSoftMatcherModel = (m: ModelType): m is SoftMatcherModel => m.config.model_type === 'soft_matcher';
export const isYoloeModel = (m: ModelType): m is YoloeModel => m.config.model_type === 'yoloe';
