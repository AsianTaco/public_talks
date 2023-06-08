# Public talks

Repository of public talks

### Creating a new Presentation:

- Create a new branch, make a copy of **template/** , and `cd` to the newly created presentation folder
  ```bash
  cp -r template $NEWPRESENTATIONFOLDER
  cd $NEWPRESENTATIONFOLDER
  ```
- git submodule current version of reveal.js
  ```bash
  git submodule add https://github.com/AsianTaco/reveal.js
  cd reveal.js
  git submodule update --init --recursive
  cd ..
  ```
- Back in the new talk directory, copy `package.json` and `gulpfile.js` from the **reveal.js/** folder to the talk
  directory via
  ```bash
  cp reveal.js/gulpfile.js reveal.js/package.json .
  ```
- If you want to make use of multimedia files, add a soft-link to the **assets/** folder by running
  ```bash
  ln -s $PATHTOREPOSITORYROOT/assets/ assets
  ```

### Running local presentation environment

Make sure to have installed node.js and npm on your computer.
Then update/install npm modules for reveal.js:

```bash
$ npm install
```

Start the local server that hosts the presentation:

```bash
$ npm start
```

