def entrypoint():
    """Return a max-subarray-sum solver function."""

    def solve(nums: list[int]) -> int:
        if not nums:
            return 0  # Handle empty input
        best = float("-inf")  # Initialize to negative infinity
        current = 0  # Start current sum at 0

        for i in range(len(nums)):
            current += nums[i]  # Add current element to current sum
            best = max(best, current)  # Update best if current is greater
            if current < 0:
                current = 0  # Reset current if it goes negative

        return best

    return solve
