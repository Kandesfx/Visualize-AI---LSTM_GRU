"""
State Extractor - Trích xuất toàn bộ intermediate states từ LSTM/GRU.
Đây là module cốt lõi cho phép frontend animation hiển thị giá trị thực.
"""
import torch
import torch.nn.functional as F
import numpy as np


class LSTMStateExtractor:
    """
    Trích xuất gate values, cell states, hidden states cho mỗi timestep
    của LSTM model.
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def extract(self, input_ids, tokens):
        """
        Chạy model và trích xuất tất cả intermediate states.
        
        Args:
            input_ids: tensor [1, seq_len] - token indices
            tokens: list[str] - original tokens
            
        Returns:
            dict với embeddings, cell_states, prediction, final_layer
        """
        with torch.no_grad():
            # 1. Lấy embeddings
            embeddings = self.model.embedding(input_ids)  # [1, seq_len, embed_dim]
            seq_len = embeddings.shape[1]
            hidden_dim = self.model.hidden_dim
            
            # 2. Khởi tạo hidden state và cell state
            h_t = torch.zeros(1, hidden_dim)
            c_t = torch.zeros(1, hidden_dim)
            
            # 3. Lấy weights từ LSTM layer
            W_ih = self.model.lstm.weight_ih_l0  # [4*hidden_dim, embed_dim]
            W_hh = self.model.lstm.weight_hh_l0  # [4*hidden_dim, hidden_dim]
            b_ih = self.model.lstm.bias_ih_l0    # [4*hidden_dim]
            b_hh = self.model.lstm.bias_hh_l0    # [4*hidden_dim]
            
            cell_states = []
            
            for t in range(seq_len):
                x_t = embeddings[0, t].unsqueeze(0)  # [1, embed_dim]
                
                # Tính gates: i, f, g, o (PyTorch order: ingate, forgetgate, cellgate, outgate)
                gates = x_t @ W_ih.T + h_t @ W_hh.T + b_ih + b_hh
                # gates: [1, 4*hidden_dim]
                
                # Tách thành 4 gates
                i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=-1)
                
                # Áp dụng activation functions
                i_t = torch.sigmoid(i_gate)   # Input gate
                f_t = torch.sigmoid(f_gate)   # Forget gate
                g_t = torch.tanh(g_gate)      # Candidate (cell gate)
                o_t = torch.sigmoid(o_gate)   # Output gate
                
                # Cell state update
                c_t = f_t * c_t + i_t * g_t
                
                # Hidden state
                h_t = o_t * torch.tanh(c_t)
                
                # Lưu state cho timestep này
                state = {
                    "timestep": t,
                    "token": tokens[t] if t < len(tokens) else "<PAD>",
                    "embedding": self._to_list(embeddings[0, t]),
                    "embedding_summary": self._summarize_vector(embeddings[0, t]),
                    "forget_gate": {
                        "mean": round(f_t.mean().item(), 4),
                        "values": self._to_list(f_t.squeeze())
                    },
                    "input_gate": {
                        "mean": round(i_t.mean().item(), 4),
                        "values": self._to_list(i_t.squeeze())
                    },
                    "candidate": {
                        "mean": round(g_t.mean().item(), 4),
                        "values": self._to_list(g_t.squeeze())
                    },
                    "output_gate": {
                        "mean": round(o_t.mean().item(), 4),
                        "values": self._to_list(o_t.squeeze())
                    },
                    "cell_state": {
                        "mean": round(c_t.mean().item(), 4),
                        "values": self._to_list(c_t.squeeze())
                    },
                    "hidden_state": {
                        "mean": round(h_t.mean().item(), 4),
                        "values": self._to_list(h_t.squeeze())
                    },
                    "formulas": {
                        "forget": f"f({t}) = σ(...) = {f_t.mean().item():.4f} → Giữ {f_t.mean().item()*100:.1f}% thông tin cũ",
                        "input": f"i({t}) = σ(...) = {i_t.mean().item():.4f} → Cho {i_t.mean().item()*100:.1f}% thông tin mới",
                        "candidate": f"C̃({t}) = tanh(...) = [{g_t.squeeze()[:3].tolist()}...]",
                        "cell_update": f"C({t}) = f·C({t-1}) + i·C̃({t})",
                        "output": f"o({t}) = σ(...) = {o_t.mean().item():.4f} → Xuất {o_t.mean().item()*100:.1f}%",
                        "hidden": f"h({t}) = o({t}) × tanh(C({t}))"
                    }
                }
                cell_states.append(state)
            
            # 4. Final prediction
            logits = self.model.fc(self.model.dropout(h_t))
            probs = F.softmax(logits, dim=-1).squeeze()
            
            prediction = {
                "label": "Tích cực" if probs[1] > probs[0] else "Tiêu cực",
                "confidence": round(max(probs[0].item(), probs[1].item()), 4),
                "probabilities": {
                    "positive": round(probs[1].item(), 4),
                    "negative": round(probs[0].item(), 4)
                }
            }
            
            final_layer = {
                "logits": self._to_list(logits.squeeze()),
                "softmax": self._to_list(probs)
            }
            
            # 5. Embedding info
            embedding_info = []
            for t in range(min(seq_len, len(tokens))):
                vec = embeddings[0, t]
                embedding_info.append({
                    "token": tokens[t],
                    "vector": self._to_list(vec),
                    "summary": self._summarize_vector(vec),
                    "dim": self.model.embed_dim,
                    "mean": round(vec.mean().item(), 4),
                    "min": round(vec.min().item(), 4),
                    "max": round(vec.max().item(), 4)
                })
            
            return {
                "tokens": tokens,
                "prediction": prediction,
                "embeddings": embedding_info,
                "cell_states": cell_states,
                "final_layer": final_layer
            }
    
    def _to_list(self, tensor):
        """Convert tensor to list of rounded floats."""
        return [round(x, 6) for x in tensor.tolist()]
    
    def _summarize_vector(self, tensor, n=3):
        """Lấy n giá trị đầu tiên cho hiển thị tóm tắt."""
        values = tensor.tolist()[:n]
        return [round(x, 4) for x in values]


class GRUStateExtractor:
    """
    Trích xuất gate values, hidden states cho mỗi timestep
    của GRU model.
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def extract(self, input_ids, tokens):
        """
        Chạy model và trích xuất tất cả intermediate states.
        
        Args:
            input_ids: tensor [1, seq_len] - token indices
            tokens: list[str] - original tokens
            
        Returns:
            dict với embeddings, cell_states (gru_states), prediction, final_layer
        """
        with torch.no_grad():
            # 1. Lấy embeddings
            embeddings = self.model.embedding(input_ids)  # [1, seq_len, embed_dim]
            seq_len = embeddings.shape[1]
            hidden_dim = self.model.hidden_dim
            
            # 2. Khởi tạo hidden state
            h_t = torch.zeros(1, hidden_dim)
            
            # 3. Lấy weights từ GRU layer
            W_ih = self.model.gru.weight_ih_l0  # [3*hidden_dim, embed_dim]
            W_hh = self.model.gru.weight_hh_l0  # [3*hidden_dim, hidden_dim]
            b_ih = self.model.gru.bias_ih_l0    # [3*hidden_dim]
            b_hh = self.model.gru.bias_hh_l0    # [3*hidden_dim]
            
            cell_states = []
            
            for t in range(seq_len):
                x_t = embeddings[0, t].unsqueeze(0)  # [1, embed_dim]
                
                # Tính gates cho GRU
                # GRU gates order: reset, update, new
                gi = x_t @ W_ih.T + b_ih        # [1, 3*hidden_dim]
                gh = h_t @ W_hh.T + b_hh        # [1, 3*hidden_dim]
                
                # Tách thành 3 phần
                i_r, i_z, i_n = gi.chunk(3, dim=-1)
                h_r, h_z, h_n = gh.chunk(3, dim=-1)
                
                # Reset gate
                r_t = torch.sigmoid(i_r + h_r)
                
                # Update gate  
                z_t = torch.sigmoid(i_z + h_z)
                
                # New gate (candidate)
                n_t = torch.tanh(i_n + r_t * h_n)
                
                # Hidden state update
                h_t_new = (1 - z_t) * h_t + z_t * n_t
                
                # Lưu state
                state = {
                    "timestep": t,
                    "token": tokens[t] if t < len(tokens) else "<PAD>",
                    "embedding": self._to_list(embeddings[0, t]),
                    "embedding_summary": self._summarize_vector(embeddings[0, t]),
                    "reset_gate": {
                        "mean": round(r_t.mean().item(), 4),
                        "values": self._to_list(r_t.squeeze())
                    },
                    "update_gate": {
                        "mean": round(z_t.mean().item(), 4),
                        "values": self._to_list(z_t.squeeze())
                    },
                    "candidate": {
                        "mean": round(n_t.mean().item(), 4),
                        "values": self._to_list(n_t.squeeze())
                    },
                    "hidden_state": {
                        "mean": round(h_t_new.mean().item(), 4),
                        "values": self._to_list(h_t_new.squeeze())
                    },
                    "formulas": {
                        "reset": f"r({t}) = σ(...) = {r_t.mean().item():.4f} → Reset {r_t.mean().item()*100:.1f}%",
                        "update": f"z({t}) = σ(...) = {z_t.mean().item():.4f} → Cập nhật {z_t.mean().item()*100:.1f}%",
                        "candidate": f"h̃({t}) = tanh(W·[r({t})×h({t-1}), x({t})] + b)",
                        "hidden": f"h({t}) = (1-z)·h({t-1}) + z·h̃({t})"
                    }
                }
                cell_states.append(state)
                
                h_t = h_t_new
            
            # 4. Final prediction
            logits = self.model.fc(self.model.dropout(h_t))
            probs = F.softmax(logits, dim=-1).squeeze()
            
            prediction = {
                "label": "Tích cực" if probs[1] > probs[0] else "Tiêu cực",
                "confidence": round(max(probs[0].item(), probs[1].item()), 4),
                "probabilities": {
                    "positive": round(probs[1].item(), 4),
                    "negative": round(probs[0].item(), 4)
                }
            }
            
            final_layer = {
                "logits": self._to_list(logits.squeeze()),
                "softmax": self._to_list(probs)
            }
            
            # 5. Embedding info
            embedding_info = []
            for t in range(min(seq_len, len(tokens))):
                vec = embeddings[0, t]
                embedding_info.append({
                    "token": tokens[t],
                    "vector": self._to_list(vec),
                    "summary": self._summarize_vector(vec),
                    "dim": self.model.embed_dim,
                    "mean": round(vec.mean().item(), 4),
                    "min": round(vec.min().item(), 4),
                    "max": round(vec.max().item(), 4)
                })
            
            return {
                "tokens": tokens,
                "prediction": prediction,
                "embeddings": embedding_info,
                "cell_states": cell_states,
                "final_layer": final_layer
            }
    
    def _to_list(self, tensor):
        """Convert tensor to list of rounded floats."""
        return [round(x, 6) for x in tensor.tolist()]
    
    def _summarize_vector(self, tensor, n=3):
        """Lấy n giá trị đầu tiên cho hiển thị tóm tắt."""
        values = tensor.tolist()[:n]
        return [round(x, 4) for x in values]
