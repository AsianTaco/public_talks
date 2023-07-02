# Public talks

Repository of public talks

## Creating a new Presentation:

### Step 1

Create a new branch, make a copy of **template/** , and `cd` to the newly created presentation folder
  ```bash
  cp -r template $NEWPRESENTATIONFOLDER
  cd $NEWPRESENTATIONFOLDER
  ```

### Step 2

#### Option 1

If you want an individual reveal.js instance for the talk, then use git to create a submodule of current (or any other) version of reveal.js

  ```bash
  git submodule add https://github.com/AsianTaco/reveal.js
  cd reveal.js
  git submodule update --init --recursive
  cd ..
  ```

#### Option 2

If you don't care too much about the specific version of reveal.js, then it suffices to use the already initialised
submodule of reveal.js in the root of this repository. To make it findable for the talk create a soft-link to the installation via

  ```bash
  ln -s $PATHTOREPOSITORYROOT/reveal.js/ reveal.js
  cd $PATHTOREPOSITORYROOT/reveal.js/
  git submodule update --init --recursive
  cd -
  ```

### Step 3

Back in the new talk directory, copy `package.json` and `gulpfile.js` from the **reveal.js/** folder (be careful to check that it's the correct relative path) to the talk
  directory via
  ```bash
  cp $PATHTOREVEALJS/reveal.js/gulpfile.js $PATHTOREVEALJS/reveal.js/package.json .
  ```

### Step 4 (Optional)

If you want to make use of multimedia files, add a soft-link to the **assets/** folder by running
  ```bash
  ln -s $PATHTOREPOSITORYROOT/assets/ assets
  ```

## Running local presentation environment

Make sure to have installed [node.js](https://nodejs.org/en) and [npm](https://www.npmjs.com/) on your computer.
Then update/install npm modules for reveal.js:

```bash
$ npm install
```

Start the local server that hosts the presentation:

```bash
$ npm start
```

