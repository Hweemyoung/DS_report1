from layers import *

class CharacterEmbeddings(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, pretrained_embeddings=None, initial_weights=None):
        '''
        :param num_embeddings: int
        :param embedding_dim: int
        :param pretrained_embeddings: 2-d LongTensor
        :param initial_weights: 2-d LongTensor
        '''
        super(CharacterEmbeddings, self).__init__()
        if pretrained_embeddings:
            self.character_embeddings = nn.Embedding.from_pretrained(
                embeddings=pretrained_embeddings, freeze=True)
        elif initial_weights:
            self.character_embeddings = nn.Embedding(num_embeddings=num_embeddings,
                                                     embedding_dim=embedding_dim,
                                                     _weight=initial_weights)
        else:
            self.character_embeddings = nn.Embedding(num_embeddings=num_embeddings,
                                                     embedding_dim=embedding_dim)

    def forward(self, input_indices):
        '''
        :param input_indices: 2-d LongTensor of arbitrary shape containing the indices to extract
            Size([batch_size, max_input_length])
        :return: 3-d LongTensor
            Size([batch_size, max_input_length, embedding_dim])
        '''
        return self.character_embeddings(input_indices)