# struct NnueAccumulator {
#     // Two vectors of size M. v[0] for white's, and v[1] for black's perspectives.
#     float v[2][M];
#
#     // This will be utilised in later code snippets to make the access less verbose
#     float* operator[](Color perspective) {
#         return v[perspective];
#     }
# };

from typing import List

class NnueAccumulator:
    def __init__(self, size: int):
        # Initialize two lists for white and black perspectives
        self.v: List[List[int]] = [[0] for _ in range(size)]

    def __getitem__(self, player: int) -> List[int]:
        return self.v[player]

# void refresh_accumulator(
#     const LinearLayer&      layer,            // this will always be L_0
#     NnueAccumulator&        new_acc,          // storage for the result
#     const std::vector<int>& active_features,  // the indices of features that are active for this position
#     Color                   perspective       // the perspective to refresh
# ) {
#     // First we copy the layer bias, that's our starting point
#     for (int i = 0; i < M; ++i) {
#         new_acc[perspective][i] = layer.bias[i];
#     }
#
#     // Then we just accumulate all the columns for the active features. That's what accumulators do!
#     for (int a : active_features) {
#         for (int i = 0; i < M; ++i) {
#             new_acc[perspective][i] += layer.weight[a][i];
#         }
#     }
# }

def refresh_accumulator(layer, new_acc, active_features, perspective) -> None:
    # Copy the bias values to the accumulator
    for i in range(len(layer.bias)):
        new_acc[perspective][i] = layer.bias[i]

    # Accumulate the weights for the active features
    for a in active_features:
        for i in range(len(layer.weight[a])):
            new_acc[perspective][i] += layer.weight[a][i]

# void update_accumulator(
#     const LinearLayer&      layer,            // this will always be L_0
#     NnueAccumulator&        new_acc,          // it's nice to have already provided storage for
#                                               // the new accumulator. Relevant parts will be overwritten
#     const NNueAccumulator&  prev_acc,         // the previous accumulator, the one we're reusing
#     const std::vector<int>& removed_features, // the indices of features that were removed
#     const std::vector<int>& added_features,   // the indices of features that were added
#     Color                   perspective       // the perspective to update, remember we have two,
#                                               // they have separate feature lists, and it even may happen
#                                               // that one is updated while the other needs a full refresh
# ) {
#     // First we copy the previous values, that's our starting point
#     for (int i = 0; i < M; ++i) {
#         new_acc[perspective][i] = prev_acc[perspective][i];
#     }
#
#     // Then we subtract the weights of the removed features
#     for (int r : removed_features) {
#         for (int i = 0; i < M; ++i) {
#             // Just subtract r-th column
#             new_acc[perspective][i] -= layer.weight[r][i];
#         }
#     }
#
#     // Similar for the added features, but add instead of subtracting
#     for (int a : added_features) {
#         for (int i = 0; i < M; ++i) {
#             new_acc[perspective][i] += layer.weight[a][i];
#         }
#     }
# }

def update_accumulator(layer, new_acc, prev_acc, removed_features, added_features, perspective) -> None:
    # Copy the previous values to the accumulator
    for i in range(len(prev_acc[perspective])):
        new_acc[perspective][i] = prev_acc[perspective][i]

    # Subtract the weights of the removed features
    for r in removed_features:
        for i in range(len(layer.weight[r])):
            new_acc[perspective][i] -= layer.weight[r][i]

    # Add the weights of the added features
    for a in added_features:
        for i in range(len(layer.weight[a])):
            new_acc[perspective][i] += layer.weight[a][i]
