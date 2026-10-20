"""Peça 1 + Peça 3 do MYELIN: a Sinapse Rápida de posto baixo e a Mielinização.

    ĥ_t^(k) = f_k(h_t^(k-1)) + B_k (A_k^T h_t^(k-1))          (Peça 1)
    B_k     <- γ B_k + η · r_t (A_k^T h_t^(k-1))^T              (Peça 3)

A_k é uma projeção fixa (aleatória, colunas ortonormais) e B_k começa em zero,
de modo que no início da sessão a rede é idêntica ao modelo de fábrica.
Só B_k muda durante a inferência, por uma regra local (Delta Rule / Hebbiana
modulada por erro), sem autograd e sem backprop pelas camadas profundas.
"""

from __future__ import annotations

import torch
from torch import nn


class FastSynapse(nn.Module):
    """Adaptador ΔW_k(h) = B (A^T h) com tamanho fixo, atualizado em tempo de teste.

    Args:
        d_in: dimensão do estado de entrada h^(k-1).
        d_out: dimensão do estado de saída da camada rasa ĥ^(k).
        rank: posto r da sinapse (r << d).
        eta: taxa de aprendizado η da Mielinização.
        gamma: decaimento homeostático γ aplicado a cada consolidação.
        normalized: se True, usa a Delta Rule normalizada (NLMS), dividindo o
            passo por ||A^T h||². Isso torna η adimensional e estável para
            qualquer escala de h (estados ocultos de LLMs reais têm normas
            muito variáveis). Com False, aplica a equação da especificação
            literalmente.
        seed: semente da projeção fixa A.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        rank: int = 16,
        eta: float = 0.5,
        gamma: float = 0.995,
        normalized: bool = True,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma deve estar em (0, 1], recebido {gamma}")
        if rank > d_in:
            raise ValueError(f"rank ({rank}) não pode exceder d_in ({d_in})")

        self.d_in, self.d_out, self.rank = d_in, d_out, rank
        self.eta, self.gamma, self.normalized = eta, gamma, normalized

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        A, _ = torch.linalg.qr(torch.randn(d_in, rank, generator=generator))

        # Buffers, não Parameters: a sinapse nunca entra no grafo do autograd.
        self.register_buffer("A", A)
        self.register_buffer("B", torch.zeros(d_out, rank))

    def forward(self, h_in: torch.Tensor) -> torch.Tensor:
        """Retorna ΔW_k(h) = B (A^T h) para h de forma (..., d_in)."""
        return (h_in @ self.A) @ self.B.T

    @torch.no_grad()
    def decay(self) -> None:
        """Homeostase sozinha: B <- γB (para decair a cada token, e não só a cada consolidação)."""
        self.B.mul_(self.gamma)

    @torch.no_grad()
    def consolidate(
        self,
        h_in: torch.Tensor,
        residual: torch.Tensor,
        mask: torch.Tensor | None = None,
        decay: bool = True,
    ) -> None:
        """Mielinização: B <- γB + η · média_t[ r_t z_t^T ], com z_t = A^T h_t.

        Args:
            h_in: estados de entrada h^(k-1), forma (..., d_in).
            residual: resíduo r_t = Norm(h^(L)) - Norm(ĥ^(k)), forma (..., d_out).
            mask: booleano (...,) indicando quais tokens tiveram estresse alto
                (só esses ensinam a sinapse). None = todos.
            decay: aplica γ junto com a consolidação. Use False quando o
                decaimento já é aplicado a cada token com `decay()`.
        """
        z = h_in.reshape(-1, self.d_in) @ self.A
        r = residual.reshape(-1, self.d_out)
        if mask is not None:
            keep = mask.reshape(-1)
            z, r = z[keep], r[keep]
        if z.shape[0] == 0:
            return

        if self.normalized:
            r = r / (z.pow(2).sum(-1, keepdim=True) + 1e-6)

        if decay:
            self.B.mul_(self.gamma)
        self.B.add_(r.T @ z, alpha=self.eta / z.shape[0])

    @torch.no_grad()
    def reset(self) -> None:
        """Volta ao modelo de fábrica (B = 0), ex.: no início de uma nova sessão."""
        self.B.zero_()

    def memory_bytes(self) -> int:
        """Memória ocupada pela sinapse. Fixa: não cresce com o uso."""
        return sum(buf.numel() * buf.element_size() for buf in (self.A, self.B))

    def extra_repr(self) -> str:
        return (
            f"d_in={self.d_in}, d_out={self.d_out}, rank={self.rank}, "
            f"eta={self.eta}, gamma={self.gamma}, normalized={self.normalized}"
        )
