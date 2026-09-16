def entrypoint():
    """Return a max-subarray-sum solver function."""

    def solve(nums: list[int]) -> int:
        if not nums:
            raise ValueError("Input list cannot be empty.")

        current_sum = max_sum = nums[
            0
        ]  # Initialize to the first element to handle negatives

        for num in nums[1:]:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)

        return max_sum

    return solve
