def entrypoint():
    """Return a max-subarray-sum solver function."""

    def solve(nums: list[int]) -> int:
        if not nums:
            raise ValueError("Input list cannot be empty.")  # Input validation
        best = nums[0]  # Initialize to first element to handle all-negative cases
        current = 0
        for num in nums:
            current += num
            if current > best:
                best = current
            if current < 0:
                current = 0  # Reset current sum if it drops below zero
        return best

    return solve
