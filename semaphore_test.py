import asyncio
import random

# Dummy function to simulate cmd execution
async def run_command(cmd, input_number):
    print(f"Starting {cmd}({input_number})")
    await asyncio.sleep(5)  # Simulate the command's execution time
    print(f"Finished {cmd}({input_number})")

# Function to manage sequential execution for a single input
async def process_input(input_number, cmd1_semaphore, cmd2_semaphore, cmd3_semaphore):
    async with cmd1_semaphore:
        await run_command("cmd1", input_number)
    async with cmd2_semaphore:
        await run_command("cmd2", input_number)
    async with cmd3_semaphore:
        await run_command("cmd3", input_number)

# Main function to handle the processing pipeline
async def main(input_numbers):
    # Semaphores to manage concurrency
    cmd1_semaphore = asyncio.Semaphore(1)  # Only one cmd1 runs at a time
    cmd2_semaphore = asyncio.Semaphore(2)  # cmd2 can overlap with cmd1 of the next number
    cmd3_semaphore = asyncio.Semaphore(3)  # cmd3 has the most concurrency

    # Schedule all tasks
    tasks = [
        process_input(num, cmd1_semaphore, cmd2_semaphore, cmd3_semaphore)
        for num in input_numbers
    ]
    await asyncio.gather(*tasks)

# Input data
input_numbers = [1, 2, 3, 4, 5]

# Run the main loop
asyncio.run(main(input_numbers))