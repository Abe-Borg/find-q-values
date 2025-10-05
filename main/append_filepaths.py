# Script to append est_q_values filepaths to game_settings.py

game_settings_path = 'utils/game_settings.py'

# Generate the lines to append
lines_to_append = []
for i in range(1, 101):
    line = f'est_q_values_filepath_part_{i} = base_directory / ".." / "est_q_values" / "est_q_values_part_{i}.pkl"\n'
    lines_to_append.append(line)

# Append to the file
with open(game_settings_path, 'a') as f:
    f.write('\n')  # Add a blank line before appending
    f.writelines(lines_to_append)

print(f"Appended {len(lines_to_append)} lines to {game_settings_path}")