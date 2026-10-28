# Security policy

## What this project is, security-wise

`snndsl` compiles a text description of a neural network into Python source and
into an NIR file, and optionally loads a `.pt` checkpoint. That is the whole
attack surface, and it is worth being precise about it:

- **Generated Python is executed.** `load_module()` writes a `.py` file and
  imports it. A malicious `.snn` file cannot inject arbitrary code today — the
  grammar admits expressions over tensors, not imports or calls — but the
  generated module is ordinary Python and trusts its inputs. Treat a `.snn` file
  from an untrusted source the way you would treat a Python file.
- **Checkpoints are unpickled.** `snndsl deploy --weights` and `snndsl infer`
  call `torch.load`. A `.pt` file from an untrusted source can execute code on
  load. This is a property of `torch.load`, not of this project, and it is why
  `--weights` is never required.
- **NIR files are parsed, not executed.** `nir.read` deserialises a JSON-ish
  structure; the failure mode is an exception, not code execution.

Nothing here is a network service, nothing listens on a socket, and nothing
writes outside the output directory you pass with `-o`.

## Supported versions

The latest tagged release on `main` receives fixes. Pre-1.0 there are no
long-term branches; a fix ships as a new minor version.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private
vulnerability reporting on this repository (Security → Report a vulnerability),
or email the maintainer listed on the GitHub profile.

Include:

- the version (`snndsl --version`) and the output of `snndsl doctor`;
- the smallest file or command that triggers it;
- what an attacker gains — code execution, file write, denial of service.

Expect an acknowledgement within a week. If the report is about the generated
code executing something it should not, that is in scope and treated as high
severity. If it is about `torch.load` doing what `torch.load` does, it will be
documented here rather than fixed.
