import torch
import numpy as np
from copy import deepcopy
import open_clip
from shapiq import Game

class OpenCLIPGame(Game):
    """
    A custom interface strictly targeting OpenCLIP models (like DermLIP, CoCa, BiomedCLIP)
    """
    def __init__(self, model, image_processor, text_tokenizer, input_image, input_text, patch_size=16, batch_size=64, verbose=False):
        self.model = model
        self.processor = image_processor
        self.text_tokenizer = text_tokenizer
        
        self.input_image = input_image
        self.input_text = input_text
        self.batch_size = batch_size

        # Find padding token dynamically 
        self.pad_token_id = self._find_pad_token()
        print("PAD_TOKEN: ", self.pad_token_id)

        # Run processor once to infer sequence sizing
        self.inputs = self._processor_function([input_image], [input_text])

        # Safely extract image parameters (OpenCLIP often uses `model.visual.image_size`)
        if hasattr(self.model, 'visual'):
            if hasattr(self.model.visual, 'image_size'):
                self.image_size = self.model.visual.image_size[0] if isinstance(self.model.visual.image_size, tuple) else self.model.visual.image_size
            else:
                self.image_size = 224 # fallback
        else:
            self.image_size = 224
        
        self.patch_size = patch_size
        self.n_channels = 3
        self.grid_size = self.image_size // self.patch_size
        self.n_players_image = int(self.image_size / self.patch_size) ** 2 
        
        # In open_clip, BOS and EOS wrap the tokens.
        # We find the number of tokens by excluding padding and special tokens. 
        text_tensor = self.inputs[1][0]
        # Count non-pad tokens subtracting 2 for BOS and EOS.
        self.n_players_text = (text_tensor != self.pad_token_id).sum().item() - 2 
        # Handle Edge cases where n_players_text < 1
        if self.n_players_text < 1:
            self.n_players_text = 1

        self.text_context_length = self.inputs[1].shape[-1]
        self.device = next(self.model.parameters()).device

        # get the normalization value
        coalitions = np.zeros((2, self.n_players_image + self.n_players_text), dtype=bool)
        coalitions[1, :] = True
        game_output = self.value_function(coalitions=coalitions)
        self.empty_value = float(game_output[0])
        self.full_value = float(game_output[1])

        if verbose:
            print(f"Similarly of the Image and Text: {self.full_value} (empty_value={self.empty_value})")

        super().__init__(
            n_players=self.n_players_image + self.n_players_text,
            normalize=True,
            normalization_value=self.empty_value,
            verbose=False
        )

    def _find_pad_token(self):
        """Attempts to find the padding/empty token dynamically for open_clip tokenizers"""
        try:
            # Often padding is 0 in OpenCLIP
            if hasattr(self.text_tokenizer, 'vocab'):
                return self.text_tokenizer.vocab.get('<pad>', 0)
            
            # Alternatively encode empty string and find the token filling the tensor
            dummy_encoding = self.text_tokenizer([""])[0]
            # Usually the last token in the fixed tensor length is the padding token
            return dummy_encoding[-1].item()
        except:
            return 0

    def _processor_function(self, input_image, input_text):
        """
        Input: list of images of length N, list of texts of length M.
        Output: a list of processed inputs as [pixel_values_tensor, text_tensor]
        """
        text = self.text_tokenizer(input_text)
        image = torch.stack([self.processor(i) for i in input_image])
        return [image, text]

    def value_function(self, coalitions, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size 
        n_coalitions = coalitions.shape[0]
        coalitions_image = torch.from_numpy(coalitions[:, :self.n_players_image])
        coalitions_text = torch.from_numpy(coalitions[:, self.n_players_image:])
        
        # [n_coalitions, text_context_length] -> pad with ones around the active tokens
        text_binary_masks = torch.cat(
            (torch.ones(n_coalitions, 1), coalitions_text, 
             torch.ones(n_coalitions, 1), torch.zeros(n_coalitions, self.text_context_length - self.n_players_text - 2)), 
            axis=1
        ).int()
        
        image_binary_masks = self._generate_image_binary_mask(coalitions_image)
        inputs_original = self._processor_function([self.input_image] * batch_size, [self.input_text] * batch_size)

        batch_iters = n_coalitions // batch_size
        batch_left = n_coalitions % batch_size
        coalitions_outputs = []
        for batch_index in range(batch_iters + 1):
            if batch_index < batch_iters:
                inputs = deepcopy(inputs_original)
                inputs[0] = (inputs[0] * image_binary_masks[(batch_index * batch_size):((batch_index + 1) * batch_size)]).to(self.device)
                
                inputs[1] = (inputs[1] * text_binary_masks[(batch_index * batch_size):((batch_index + 1) * batch_size)] +\
                             self.pad_token_id * (1 - text_binary_masks[(batch_index * batch_size):((batch_index + 1) * batch_size)])).to(self.device)
            elif batch_left > 0: 
                inputs = self._processor_function([self.input_image]*batch_left, [self.input_text]*batch_left)
                inputs[0] = (inputs[0] * image_binary_masks[(batch_index * batch_size):(batch_index * batch_size + batch_left)]).to(self.device)
                inputs[1] = (inputs[1] * text_binary_masks[(batch_index * batch_size):(batch_index * batch_size + batch_left)] +\
                             self.pad_token_id * (1 - text_binary_masks[(batch_index * batch_size):(batch_index * batch_size + batch_left)])).to(self.device)
            else:
                break 
            
            with torch.no_grad():
                # Handling custom OpenCLIP forwarding structure 
                image_features, text_features, logit_scale = self.model(*inputs)
                
                # Normalize exactly like open_clip does 
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                if logit_scale is not None:
                     logits_per_image = logit_scale * image_features @ text_features.T
                else:
                     logits_per_image = image_features @ text_features.T
                     
            outputs = torch.diagonal(logits_per_image).cpu()
            coalitions_outputs.append(outputs)
        coalitions_outputs = torch.concat(coalitions_outputs)
        return coalitions_outputs.numpy()

    def value_function_crossmodal(self, coalitions_image, coalitions_text, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size 
        n_coalitions_image = coalitions_image.shape[0]
        n_coalitions_text = coalitions_text.shape[0]

        text_binary_masks = torch.cat(
            (torch.ones(n_coalitions_text, 1), torch.from_numpy(coalitions_text), 
             torch.ones(n_coalitions_text, 1), torch.zeros(n_coalitions_text, self.text_context_length - self.n_players_text - 2)), 
            axis=1
        ).int()
        
        image_binary_masks = self._generate_image_binary_mask(torch.from_numpy(coalitions_image))
        inputs_original = self._processor_function([self.input_image] * batch_size, [self.input_text] * batch_size)

        batch_iters_image = n_coalitions_image // batch_size
        batch_iters_text = n_coalitions_text // batch_size
        batch_left_image = n_coalitions_image % batch_size
        batch_left_text = n_coalitions_text % batch_size
        if batch_left_text > 0: 
            inputs_left_text = self._processor_function([self.input_image] * batch_size, [self.input_text] * batch_left_text)

        coalitions_outputs = []
        for batch_index_image in range(batch_iters_image + 1):
            coalitions_outputs_image = []
            if batch_index_image < batch_iters_image:
                inputs_image = deepcopy(inputs_original)
                inputs_image[0] = (inputs_image[0] *\
                              image_binary_masks[(batch_index_image * batch_size):((batch_index_image + 1) * batch_size)]).to(self.device)
            elif batch_left_image > 0:
                inputs_image = self._processor_function([self.input_image] * batch_left_image, [self.input_text] * batch_size)
                inputs_image[0] = (inputs_image[0] *\
                              image_binary_masks[(batch_index_image * batch_size):(batch_index_image * batch_size + batch_left_image)]).to(self.device)
            else:
                break 
            for batch_index_text in range(batch_iters_text + 1):
                if batch_index_text < batch_iters_text:
                    inputs = deepcopy(inputs_image)
                    inputs[1] = (inputs[1] *\
                                  text_binary_masks[(batch_index_text * batch_size):((batch_index_text + 1) * batch_size)] +\
                                    self.pad_token_id * (1 - text_binary_masks[(batch_index_text * batch_size):((batch_index_text + 1) * batch_size)])).to(self.device)
                elif batch_left_text > 0 and batch_index_image < batch_iters_image: 
                    inputs = deepcopy(inputs_left_text)
                    inputs[0] = (inputs[0] *\
                                image_binary_masks[(batch_index_image * batch_size):((batch_index_image + 1) * batch_size)]).to(self.device)
                    inputs[1] = (inputs[1] *\
                                  text_binary_masks[(batch_index_text * batch_size):(batch_index_text * batch_size + batch_left_text)] +\
                                    self.pad_token_id * (1 - text_binary_masks[(batch_index_text * batch_size):(batch_index_text * batch_size + batch_left_text)])).to(self.device)                
                elif batch_left_text > 0 and batch_left_image > 0: 
                    inputs = self._processor_function([self.input_image] * batch_left_image, [self.input_text] * batch_left_text)
                    inputs[0] = (inputs[0] *\
                                image_binary_masks[(batch_index_image * batch_size):(batch_index_image * batch_size + batch_left_image)]).to(self.device)
                    inputs[1] = (inputs[1] *\
                                  text_binary_masks[(batch_index_text * batch_size):(batch_index_text * batch_size + batch_left_text)] +\
                                    self.pad_token_id * (1 - text_binary_masks[(batch_index_text * batch_size):(batch_index_text * batch_size + batch_left_text)])).to(self.device)            
                else:
                    break
                with torch.no_grad():
                    image_features, text_features, logit_scale = self.model(*inputs)
                    
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)

                    if logit_scale is not None:
                        logits_per_image = logit_scale * image_features @ text_features.T
                    else:
                        logits_per_image = image_features @ text_features.T
                        
                outputs = logits_per_image.cpu() # In crossmodal we don't strictly diagonalize until output arrangement
                coalitions_outputs_image.append(outputs)
            coalitions_outputs.append(torch.concat(coalitions_outputs_image, axis=1))
        coalitions_outputs = torch.concat(coalitions_outputs, axis=0)

        return coalitions_outputs.numpy()

    def _generate_image_binary_mask(self, coalitions):
        """
        Input: binary torch tensor
        Output: binary torch tensor
        """
        n_coalitions = coalitions.shape[0]
        binary_masks = coalitions\
            .repeat_interleave(self.patch_size**2, dim=1)\
                .reshape(n_coalitions, self.grid_size, self.grid_size, self.patch_size, self.patch_size)
        binary_masks = binary_masks\
            .permute(0, 1, 3, 2, 4)\
                .reshape(n_coalitions, self.image_size, self.image_size)
        binary_masks = binary_masks\
            .repeat((self.n_channels, 1, 1, 1))\
                .permute(1, 0, 2, 3)
        return binary_masks