import numpy

def encode_to_vector(data, model, to_list=False):     
    vectors_32 = model.encode(data, normalize_embeddings=True, convert_to_numpy=True).astype(numpy.float32)
    return vectors_32 if not to_list else vectors_32.tolist()