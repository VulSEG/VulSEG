import re

def calculate_line_scores(lines):
    lines_detail = []
    max_complexity_score = 0  # Used to track the maximum complexity score
    control_flow_keywords = ['if', 'for', 'while', 'switch', 'case', 'default', 'else if', 'break', 'continue']
    assignment_pattern = re.compile(r'.*?=.*')  # Pattern to detect assignments
    function_call_pattern = re.compile(r'\b\w+\(.*\)')  # Pattern to detect function calls

    for line_number, line in enumerate(lines, 1):
        complexity_score = 0
        is_function_definition = 'void' in line  # Simplistic check for function definition

        # Check if the line contains control flow elements
        if any(keyword in line for keyword in control_flow_keywords):
            complexity_score += 2

        # Check for assignments and function calls
        if assignment_pattern.match(line):
            complexity_score += 1

        if function_call_pattern.search(line):
            complexity_score += 2  # Higher score for function calls

        # Adjust complexity for function definitions
        if is_function_definition:
            complexity_score += 2

        lines_detail.append({"complexity_score": complexity_score})
        max_complexity_score = max(max_complexity_score, complexity_score)

    # Normalize scores from 1 to 2 and round to two decimal places
    scores = []
    for line_info in lines_detail:
        if max_complexity_score > 0:  # Prevent division by zero
            adjusted_score = 1 + (line_info["complexity_score"] / max_complexity_score)
            scores.append(round(adjusted_score, 2))  # Round to two decimal places
        else:
            scores.append(1.0)  # Default score if all lines have zero complexity

    return scores

if __name__ == '__main__':
    code_lines = ['FUN1', 'void', 'VAR1 = VAR3', 'VAR2.VAR4 = VAR1', 'FUN2(VAR2)']
    scores = calculate_line_scores(code_lines)
    print("Line Scores:", scores)

# # Example usage with your specific line format
#
#
