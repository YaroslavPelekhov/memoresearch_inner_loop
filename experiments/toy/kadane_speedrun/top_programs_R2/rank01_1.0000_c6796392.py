def entrypoint():
    """Return a max-subarray-sum solver function."""

    def solve(nums: list[int]) -> int:
        if not nums:
            return 0  # Handle empty input
        best = nums[0]  # Initialize to the first element
        current = nums[0]

        for i in range(1, len(nums)):
            current = max(nums[i], current + nums[i])  # Kadane's algorithm
            best = max(best, current)

        return best

    return solve
