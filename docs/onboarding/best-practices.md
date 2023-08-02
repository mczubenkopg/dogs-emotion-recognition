# Best practices

## Regarding code style, formatting and git tips

> For now these are good practices, but some of them should be chosen as rules
> to create a unfied style for the organization

The setup for this guide is explained on Visual Studio Code, but it applies to
all other text editors.

# Indentation

Using proper indentation helps identify separate pieces of code and greatly
improves readability. 

Set the number of spaces in a `Tab` to 4
```
Settings -> Editor -> Tab size = 4
```

Insert spaces when pressing `Tab` - this makes indentation look more consistent
on different setups
```
Settings -> Editor -> Insert spaces
```

# Line wrap

Wrapping long lines of code is a good practice, as it allows users with smaller
screens (or using an editor on half of their screen) not to scroll all over the
code to look for something. Especially useful when reading documents in
markdown.

Set word wrap to bounded - will wrap at a specified number of columns
```
Settings -> Editor -> Word wrap = bounded
```

Set number of wrap columns to 80
```
Settings -> Editor -> Word wrap column = 80
```

Do the same for markdown files
```
In settings, search for:
    @lang:markdown wrap
```

Set a ruler - line indicating where a line wrap should happen
```
Search for Editor > Rulers
Edit the settings.json file
enter 80 to create a ruler
```

### Useful extensions:

Markdown support for VS Code: [Markdown all in one](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)

Automatic hard line wrap: [Rewrap](https://marketplace.visualstudio.com/items?itemName=stkb.rewrap)

# Naming conventions

You should follow the naming convention chosen by the creators of a language, or
a publically approved convention.

Other good practices:
- Be descriptive - names that actually explain what a piece of code does
- Stick to the same convention as a team

# Markdown

Good markdown makes it easy to read both in rendered and raw form.

- Use a clear layout, using headers and lists
- Stick to the character line limit - see [line wrap](#line-wrap)
- Use code blocks for code longer than one line
  - Markdown supports syntax highlighting in code blocks, this is a preview of
    some C code.
  - ```c
        status = uefi_call_wrapper(BS->HandleProtocol, 3, IH,
          &LoadedImageProtocol, (void **)&efi_li);
        if (EFI_ERROR(status))
              panic("HandleProtocol(ImageProtocol): %" PRIxMAX, (uintmax_t)status);
    ```
  - The list of languages is very large, but some notable ones that we could use are `java`, `bash`, `css`, `shell`.
- And `inline code` for shorter pieces of code
- Use one newline between and after headers, paragraphs and lists

More extensive style guide: [Markdown style guide by Google](https://google.github.io/styleguide/docguide/style.html)

# Git

## Commit messages

- describe what the commit is about, be short and succint, but don't be overly
  specific 
- it's best to separate a commit into a few smaller ones if it encapsules
  multiple files and directories
- include a directory and file name in the message - it makes reading commit
  history easier 
- use simple verbs, like `add, fix, update`

### **Good** examples:

- `git commit -m "docs: onboarding.md: add important links section"`
- `git commit -m "netbsd: x86_64: fix import paths in EFI driver"`

### **Bad** examples:

- `git commit -m "fixed a few bugs"`
- `git commit -m "update the repository after three months of development"`
- `git commit -m "added module vim and python to core-image-minimal in build/local.conf, so that it now correctly executes the correct functions"`

## Using SSH

Allows to clone and push repos without signing in to a Github account.

This guide describes the process very clearly:
https://docs.github.com/en/authentication/connecting-to-github-with-ssh

## Signing commits using GPG

Github will put a `Verified` badge next to your commits. Signing commits with
GPG protects against identity theft/impersonation. It is a required practice in
professional work environments.

https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key

## Aliases

Git aliases can greatly speed up working with repositories. You wont't have to
spell out long words such as `checkout`, `branch`, `status`

This is a good `.gitconfig` setup:
> This file is usually located in ~/ (home directory)

```
[alias]
 ci = commit -s -S
 cia = commit --amend
 co = checkout
 br = branch
 st = status
 df = diff
 dc = diff --cached
 lg = log -p
 lol = log --graph --decorate --pretty=oneline --abbrev-commit
 lola = log --graph --decorate --pretty=oneline --abbrev-commit --all
 lolb = log --graph --decorate --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit --all --simplify-by-decoration
 hist = log --pretty=format:\"%h %ad|%s%d [%an]\" --graph --date=short
 lolg = log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit --date=relative
 ls = ls-files
 edit-unmerged = "!$EDITOR `git diff --name-only --diff-filter=U`"
 # Show files ignored by git:
 ign = ls-files -o -i --exclude-standard
 dt = difftool
 tags = for-each-ref --sort=taggerdate --format '%(refname) %(taggerdate)' refs/tags
 cp = cherry-pick
 sh = show
```

Some notable mentions:
- `co` - checkout
- `br` - branch
- `st` - status
- `df` - diff
- `lol` - graph log on current branch
- `lola` - graph log from all branches
