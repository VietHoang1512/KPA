from scipy import spatial

from qs_kpa import KeyPointAnalysis

if __name__ == "__main__":

    encoder = KeyPointAnalysis()

    print(encoder)

    inputs = [
        (
            "Assisted suicide should be a criminal offence",
            "a cure or treatment may be discovered shortly after having ended someone's life unnecessarily.",
            1,
        ),
        (
            "Assisted suicide should be a criminal offence",
            "Assisted suicide should not be allowed because many times people can still get better",
            1,
        ),
        ("Assisted suicide should be a criminal offence", "Assisted suicide is akin to killing someone", 1),
    ]

    output = encoder.encode(inputs[0], show_progress_bar=False, convert_to_numpy=True)
    print("Embedding shape", output.shape)

    output = encoder.encode(inputs, convert_to_numpy=True)
    arg_emb, pos_kp_emb, neg_kp_emb = output
    print("Positive similarity", 1 - spatial.distance.cosine(arg_emb, pos_kp_emb))
    print("Negative similarity", 1 - spatial.distance.cosine(arg_emb, neg_kp_emb))
