import ollama


def main():
    while True:
        prompt = input('\n\ngimme a prompt:')
        if prompt == 'quit':
            print('bye!')
            exit(0)
        else:
            # response = ollama.generate(model='coolguy',
            #                            prompt=prompt)
            
            # print(response['response'])

            # themodel = 'mistral'

            # themodel = 'coolguy'
            # themodel = 'werner_herzog'
            themodel = 'codegemma'
            # themodel = 'gemma:2b'
            # themodel = 'nomic-embed-text'

            stream = ollama.chat(model=themodel,
                                 options={"temperature":0.01, "top_k": 1, "top_p": 0.01},
                                 messages=[{'role': 'user', 'content': prompt}],
                                 stream=True)

            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)

if __name__ == '__main__':
    main()

