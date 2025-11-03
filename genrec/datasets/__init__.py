# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Dataset classes for GenRec."""

from genrec.datasets.AmazonReviews2014.dataset import AmazonReviews2014
from genrec.datasets.AmazonReviews2018.dataset import AmazonReviews2018

# 为了让 utils.get_dataset 能够找到，我们需要将类暴露在模块级别
__all__ = ['AmazonReviews2018', 'AmazonReviews2018']
