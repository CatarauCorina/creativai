import pickle


def load_buffer_states(i_episode=0):
    with open(f"percepts/buffer_states_{i_episode}.pickle", "rb") as output_file:
        buffer_states = pickle.load(output_file)
    el = buffer_states.sample(batch_size=10)

    return buffer_states


def main():
    load_buffer_states()


if __name__ == '__main__':
    main()
