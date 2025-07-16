import numpy as np
from collections import defaultdict


class MergeProcessor():
    def __init__(self,label,mask,k):
        self.label = label
        self.mask = mask
        self.k = k

    def merge(self):
        matrix = np.zeros(self.label.shape)
        for i, j in zip(np.nonzero(self.mask)[0], np.nonzero(self.mask)[1]):
            matrix[i, j] = self.label[i, j]
        turn = 1

        while True:
            non_zero = matrix[matrix != 0]
            unique_values = np.unique(non_zero)
            current_types = len(unique_values)
            if current_types <= self.k:
                break
            counts = defaultdict(int)
            for val in non_zero.flatten():
                counts[val] += 1
            if turn == 1:
                turn = 2
                min_count = min(counts.values())
                max_count = max(counts.values())
                remaining_candidates = [val for val, cnt in counts.items() if cnt == min_count]
            if len(remaining_candidates) == 0:
                min_count += 1
                if min_count >= max_count:
                    print("It is not possible to further reduce the number of superpixels, the current number is %d" % len(unique_values))
                    break
                remaining_candidates = [val for val, cnt in counts.items() if cnt == min_count]
                continue
            target_val = min(remaining_candidates)
            positions = np.argwhere(matrix == target_val)
            replaced = False
            for (x, y) in positions:
                neighbors = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < matrix.shape[0] and 0 <= ny < matrix.shape[1]:
                        neighbors.append((nx, ny))

                neighbor_vals = []
                for (nx, ny) in neighbors:
                    val = matrix[nx, ny]
                    if val != 0 and val != matrix[x, y]:
                        neighbor_vals.append(val)

                if not neighbor_vals:
                    continue  #no adjacent non-zero values, skip

                #count the number of occurrences of adjacent values and find the minimum value
                neighbor_counts = {val: counts[val] for val in neighbor_vals}
                min_neighbor_count = min(neighbor_counts.values())
                candidates = [val for val, cnt in neighbor_counts.items() if cnt == min_neighbor_count]
                replace_val = min(candidates)

                #replacement
                matrix[x, y] = replace_val
                replaced = True

                #update the statistics
                counts[target_val] -= 1
                counts[replace_val] += 1

                #If the original value is reset to zero, remove the key
                if counts[target_val] == 0:
                    del counts[target_val]

            #If none of the elements of the current target value can be replaced, skip and try the next minimum value
            if not replaced:
                remaining_candidates = [val for val in remaining_candidates if val != target_val]

        return matrix